from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import messagebox, ttk

from .acceptance_viewer import AcceptanceViewer
from .brush_catalog import BrushCatalog, BrushRecord, ScanResult
from .brush_cropper import BrushImageCropper
from .brush_lod import BrushLodCache
from .brush_stroke import (
    BrushStrokePlanner,
    BrushStrokeState,
    BrushStrokeTool,
    records_for_group,
)
from .gpu_batch import EMPTY_LAYER_KEY, brush_texture_key
from .gpu_edit import (
    GpuBatchResetPatch,
    GpuCellPatch,
    GpuEditBridge,
    GpuEditRequest,
    GpuResyncRequest,
    GpuSaveRequest,
    GpuStatusPatch,
    GpuSurfaceRegionPatch,
    GpuStreamPatch,
    GpuSurfaceTexturePatch,
    GpuToolState,
    GpuViewRequest,
)
from .gpu_native import NativeGpuViewportHandle, launch_gpu_preview, native_gpu_supported
from .flat_editor import ProductionFlatMapEditor
from .paths import ProjectPaths
from .production_layout import (
    ProductionChunkLayout,
    load_production_layout_cache,
    write_production_layout_cache,
)
from .production_surface import (
    ProductionSurfaceCache,
    ProductionSurfaceLiveState,
)
from .production_streaming import (
    ProductionGpuStreamingController,
    ProductionStreamingError,
)
from .production_topology import ProductionTopology
from .production_visibility import (
    ProductionVisibilityIndex,
    build_production_visibility_index,
    load_production_visibility_cache,
    write_production_visibility_cache,
)
from .sphere_map_store import SphereMapError, SphereMapSession, SphereMapStore
from .sphere_picker import (
    SphereScreenPicker,
    cap_boundary_directions,
    cell_angular_pitch,
    project_direction,
    screen_to_direction,
)
from .undo_history import UndoHistory, UndoHistoryError
from .zoom_tiers import (
    VIEWPORT_RADIUS_FACTOR,
    ZoomTierPolicy,
    cell_pixels,
    drag_radians_per_pixel,
    visible_cells_estimate,
)
from .software_render_process import (
    SoftwareGlobeRenderProcess,
    SoftwareRenderRequest,
)
from .software_globe import (
    ProjectedCellPolygon,
    average_edge_pixels,
    build_projected_cells,
    draw_shaded_sphere,
    render_textured_globe_view_ppm,
    render_textured_sphere_ppm,
    point_in_polygon,
    rotate_point,
)


class ProductionSphereEditor:
    FREQUENCY = 1004
    TILE_SIDE = 16
    MINI_GLOBE_DIAMETER = 176
    CELL_SIZE_KM = 3.0
    PARALLEL_WORKERS = 12

    def __init__(
        self,
        parent: tk.Misc,
        paths: ProjectPaths,
        *,
        main_window: bool = False,
        single_map_name: str | None = None,
        auto_open_gpu: bool = False,
    ) -> None:
        self.paths = paths
        self.main_window = bool(main_window)
        self.single_map_name = (single_map_name or "").strip() or None
        self.auto_open_gpu = bool(auto_open_gpu)
        self._auto_gpu_started = False
        self.window = parent if self.main_window else tk.Toplevel(parent)
        self.window.title(
            "六边形星球地图笔刷编辑器 v1.3.1"
            if self.main_window
            else "千万格生产球面 GPU 编辑器 v1.3.1"
        )
        self.window.geometry("1280x820" if self.main_window else "1160x760")
        self.window.minsize(1020, 680) if self.main_window else self.window.minsize(900, 620)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.bind("<Control-s>", lambda _event: self.save())
        self.window.bind("<Control-z>", lambda _event: self.undo())
        self.window.bind("<Control-y>", lambda _event: self.redo())
        self.window.bind("<Control-Shift-Z>", lambda _event: self.redo())

        self.catalog = BrushCatalog(paths.brush_root)
        self.store = SphereMapStore(paths.map_root)
        self.scan_result: ScanResult | None = None
        self.records_by_uid: dict[str, BrushRecord] = {}
        self.selected_brush_uid: str | None = None
        self.selected_brush_group: str | None = None
        self.topology: ProductionTopology | None = None
        self.picker: SphereScreenPicker | None = None
        self.undo_history = UndoHistory()
        self.stroke_planner: BrushStrokePlanner | None = None
        self.active_strokes: dict[int, BrushStrokeState] = {}
        self.stroke_rng = random.Random()
        self.local_stroke_id = 0
        self.local_stroke_active = False
        self.local_stroke_tool: BrushStrokeTool | None = None
        self.stroke_command_queue: queue.Queue[tuple[object, ...]] = queue.Queue()
        self.stroke_move_lock = threading.Lock()
        self.stroke_pending_moves: dict[int, tuple[object, ...]] = {}
        self.stroke_busy_lock = threading.Lock()
        self.stroke_busy_count = 0
        self.pending_local_save = False
        self.pending_gpu_saves: list[GpuSaveRequest] = []
        self.save_running = False
        self.save_worker: threading.Thread | None = None
        self.save_started_at = 0.0
        self.save_stage = ""
        self.save_status_second = -1
        self.close_after_save = False
        self.gpu_resync_running = False
        self.brush_prewarm_generation = 0
        self.stroke_worker_stop = threading.Event()
        self.layout: ProductionChunkLayout | None = None
        self.visibility: ProductionVisibilityIndex | None = None
        self.session: SphereMapSession | None = None
        self.controller: ProductionGpuStreamingController | None = None
        self.bridge: GpuEditBridge | None = None
        self.bridge_lod = 0
        self.embedded_gpu_starting = False
        self.embedded_gpu_active = False
        self.embedded_gpu_viewport: NativeGpuViewportHandle | None = None
        self._flat_editors: set[ProductionFlatMapEditor] = set()

        self.yaw = -0.35
        self.pitch = 0.25
        self.zoom = 1.0
        self.tier_policy = ZoomTierPolicy(initial_zoom=self.zoom)
        self.cursor_position: tuple[int, int] | None = None
        self.drag_origin: tuple[int, int, float, float, float] | None = None
        self.dragging = False
        self.surface_cache = ProductionSurfaceCache(paths.brush_root)
        self.surface_texture = None
        self.surface_signature: str | None = None
        self.surface_build_running = False
        self.surface_rebuild_requested = False
        self.surface_live_state: ProductionSurfaceLiveState | None = None
        self.surface_live_generation = 0
        self.surface_live_prepare_running = False
        self.surface_live_update_running = False
        self.surface_live_dirty_chunks: set[int] = set()
        self.surface_live_lock = threading.Lock()
        self.surface_render_running = False
        self.pending_surface_render: tuple[object, ...] | None = None
        self.surface_photo: tk.PhotoImage | None = None
        self.surface_photo_key: tuple[object, ...] | None = None
        self.surface_latest_key: tuple[object, ...] | None = None
        self.surface_render_generation = 0
        self.surface_displayed_generation = 0
        self.surface_preview_mode = False
        self.surface_renderer = SoftwareGlobeRenderProcess()
        self.mini_render_running = False
        self.mini_render_generation = 0
        self.pending_mini_render: tuple[object, ...] | None = None
        self.mini_render_job: str | None = None
        self.mini_photo: tk.PhotoImage | None = None
        self.mini_last_key: tuple[object, ...] | None = None
        self.mini_frame: ttk.LabelFrame | None = None
        self.worker_running = False
        self.view_worker_running = False
        self.pending_view: GpuViewRequest | None = None
        self.last_view_request: GpuViewRequest | None = None
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.background_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="hexplanet-background",
        )
        self.parallel_executor = ThreadPoolExecutor(
            max_workers=self.PARALLEL_WORKERS,
            thread_name_prefix="hexplanet-parallel",
        )
        threading.Thread(
            target=self._stroke_worker_loop,
            name="hexplanet-brush-stroke",
            daemon=True,
        ).start()

        self.map_name = tk.StringVar(value=self.single_map_name or "planet_production_f1004")
        self.map_choice = tk.StringVar(value="")
        self.tool = tk.StringVar(value="paint")
        self.rotation = tk.IntVar(value=0)
        self.brush_diameter = tk.IntVar(value=1)
        self.status = tk.StringVar(value="正在准备生产索引")
        self.view_load_text = tk.StringVar(value="视图：等待地图")
        self.view_load_value = tk.DoubleVar(value=0.0)
        self.summary = tk.StringVar(value="")
        self.selected_path = tk.StringVar(value="未选择笔刷")
        self.show_scale = tk.BooleanVar(value=True)
        self.scale_text = tk.StringVar(value="")
        self.tier_label = tk.StringVar(value="")
        self.undo_label = tk.StringVar(value="撤销栈：空")
        self.visible_cell_polygons: tuple[ProjectedCellPolygon, ...] = ()
        self.detail_cell_limit = 7000
        self._draw_job: str | None = None
        self._draw_due_at = 0.0
        self._final_surface_job: str | None = None

        self._build_ui()
        self.refresh_brushes()
        self.window.after(50, self._poll)
        self._start_prepare()

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(root, text="笔刷库", padding=8)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        brush_buttons = ttk.Frame(left)
        brush_buttons.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        brush_buttons.columnconfigure(0, weight=1)
        brush_buttons.columnconfigure(1, weight=1)
        ttk.Button(brush_buttons, text="刷新笔刷库", command=self.refresh_brushes).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(
            brush_buttons,
            text="图片裁剪器（输出到 art/data）",
            command=self.open_brush_cropper,
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.brush_tree = ttk.Treeview(left, show="tree", height=25)
        self.brush_tree.grid(row=1, column=0, sticky="nsew")
        self.brush_tree.bind("<<TreeviewSelect>>", self._brush_selected)
        tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.brush_tree.yview)
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.brush_tree.configure(yscrollcommand=tree_scroll.set)
        ttk.Label(left, textvariable=self.selected_path, wraplength=240).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        center = ttk.LabelFrame(
            root,
            text="整个星球地图（按住左键连续绘制，右键拖动旋转，滚轮缩放）",
            padding=6,
        )
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, background="#14191f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._paint_start)
        self.canvas.bind("<B1-Motion>", self._paint_move)
        self.canvas.bind("<ButtonRelease-1>", self._paint_end)
        self.canvas.bind("<ButtonPress-3>", self._drag_start)
        self.canvas.bind("<B3-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-3>", self._drag_end)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom(1.0))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom(-1.0))
        self.canvas.bind("<Motion>", self._track_cursor)
        self.canvas.bind("<Leave>", self._clear_cursor)
        self.surface_item = self.canvas.create_image(
            0, 0, anchor="nw", image="", tags=("surface",)
        )
        self.surface_outline_item = self.canvas.create_oval(
            0, 0, 0, 0, fill="", outline="#d0dde4", width=2,
            state=tk.HIDDEN, tags=("outline",),
        )
        self.center_message_item = self.canvas.create_text(
            0, 0, text="", fill="#e5edf2", font=("TkDefaultFont", 13),
            state=tk.HIDDEN, tags=("overlay",),
        )
        self.overlay_item = self.canvas.create_text(
            12, 12, anchor="nw", fill="#f0f5f7", text="",
            state=tk.HIDDEN, tags=("overlay",),
        )
        # The brush footprint is a geodesic disc, so under the orthographic
        # projection it is an ellipse that foreshortens toward the limb.  It is
        # drawn as a projected polygon rather than a circle so that the size the
        # user sees is the size that will actually be painted.
        self.brush_cursor_item = self.canvas.create_polygon(
            0, 0, 0, 0, 0, 0,
            fill="", outline="#ffd166", width=2,
            state=tk.HIDDEN, tags=("cursor",),
        )
        self.scale_frame = tk.Frame(
            center,
            background="#111820",
            highlightbackground="#667782",
            highlightthickness=1,
            padx=7,
            pady=5,
        )
        tk.Label(
            self.scale_frame,
            textvariable=self.scale_text,
            anchor="e",
            justify=tk.RIGHT,
            foreground="#f0f5f7",
            background="#111820",
        ).pack(fill=tk.X)
        self.scale_canvas = tk.Canvas(
            self.scale_frame,
            width=180,
            height=14,
            background="#111820",
            highlightthickness=0,
        )
        self.scale_canvas.pack()
        self.scale_bar_item = self.scale_canvas.create_line(
            20, 8, 170, 8,
            fill="#f0f5f7",
            width=2,
        )
        self.scale_left_tick_item = self.scale_canvas.create_line(
            20, 3, 20, 12,
            fill="#f0f5f7",
            width=2,
        )
        self.scale_right_tick_item = self.scale_canvas.create_line(
            170, 3, 170, 12,
            fill="#f0f5f7",
            width=2,
        )
        self.scale_frame.place(relx=1.0, rely=1.0, x=-16, y=-16, anchor="se")
        if self.main_window:
            self.mini_frame = ttk.LabelFrame(
                center,
                text="宏观实时预览",
                padding=4,
            )
            self.mini_canvas = tk.Canvas(
                self.mini_frame,
                width=self.MINI_GLOBE_DIAMETER + 8,
                height=self.MINI_GLOBE_DIAMETER + 8,
                background="#14191f",
                highlightthickness=0,
            )
            self.mini_canvas.pack()
            self.mini_image_item = self.mini_canvas.create_image(
                4,
                4,
                anchor="nw",
                image="",
            )
            self.mini_message_item = self.mini_canvas.create_text(
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                text="正在载入宏观地表…",
                fill="#d0dde4",
            )
            self.mini_reticle_item = self.mini_canvas.create_oval(
                0,
                0,
                0,
                0,
                fill="",
                outline="#ffd166",
                width=2,
            )
            self.mini_canvas.create_line(
                self.MINI_GLOBE_DIAMETER / 2,
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                self.MINI_GLOBE_DIAMETER / 2 + 8,
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                fill="#ffd166",
            )
            self.mini_canvas.create_line(
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                self.MINI_GLOBE_DIAMETER / 2,
                self.MINI_GLOBE_DIAMETER / 2 + 4,
                self.MINI_GLOBE_DIAMETER / 2 + 8,
                fill="#ffd166",
            )
            self.mini_frame.place(
                relx=1.0,
                x=-12,
                y=12,
                anchor="ne",
            )
            self._update_mini_reticle()

        right = ttk.LabelFrame(root, text="星球地图与编辑工具", padding=10)
        right.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        row = 0

        self.create_button = None
        self.map_combo = None
        self.open_button = None
        if self.single_map_name is not None:
            ttk.Label(right, text="地图").grid(row=row, column=0, sticky="w")
            row += 1
            ttk.Label(
                right,
                text="整个星球（唯一地图）",
                anchor=tk.CENTER,
                relief=tk.GROOVE,
                padding=(8, 7),
            ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
            row += 1
            ttk.Label(
                right,
                text="程序启动后会自动创建或打开这一个完整星球，不再显示16×16局部测试地图。",
                wraplength=255,
                justify=tk.LEFT,
            ).grid(row=row, column=0, sticky="w")
            row += 1
        else:
            ttk.Label(right, text="新地图名称").grid(row=row, column=0, sticky="w")
            row += 1
            ttk.Entry(right, textvariable=self.map_name, width=28).grid(
                row=row, column=0, sticky="ew", pady=(2, 5)
            )
            row += 1
            self.create_button = ttk.Button(
                right, text="创建完整 f1004 Pack 地图", command=self.create_map, state=tk.DISABLED
            )
            self.create_button.grid(row=row, column=0, sticky="ew")
            row += 1
            ttk.Label(right, text="打开兼容地图").grid(row=row, column=0, sticky="w", pady=(9, 0))
            row += 1
            self.map_combo = ttk.Combobox(right, textvariable=self.map_choice, state="readonly")
            self.map_combo.grid(row=row, column=0, sticky="ew", pady=(2, 5))
            row += 1
            self.open_button = ttk.Button(right, text="打开地图", command=self.open_map, state=tk.DISABLED)
            self.open_button.grid(row=row, column=0, sticky="ew")
            row += 1

        ttk.Separator(right).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(right, text="编辑工具").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Radiobutton(
            right, text="放置笔刷", variable=self.tool, value="paint", command=self._sync_tool
        ).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Radiobutton(
            right, text="清除格子", variable=self.tool, value="erase", command=self._sync_tool
        ).grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Radiobutton(
            right,
            text="撤销笔刷（逐格回退一次改动）",
            variable=self.tool,
            value="undo",
            command=self._sync_tool,
        ).grid(row=row, column=0, sticky="w")
        row += 1
        undo_row = ttk.Frame(right)
        undo_row.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        undo_row.columnconfigure(0, weight=1)
        undo_row.columnconfigure(1, weight=1)
        self.undo_button = ttk.Button(
            undo_row, text="撤销 Ctrl+Z", command=self.undo, state=tk.DISABLED
        )
        self.undo_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.redo_button = ttk.Button(
            undo_row, text="重做 Ctrl+Y", command=self.redo, state=tk.DISABLED
        )
        self.redo_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1
        ttk.Label(right, textvariable=self.undo_label, wraplength=250, justify=tk.LEFT).grid(
            row=row, column=0, sticky="w", pady=(3, 0)
        )
        row += 1
        ttk.Label(right, text="笔刷直径（1～500格）").grid(
            row=row, column=0, sticky="w", pady=(8, 2)
        )
        row += 1
        diameter_row = ttk.Frame(right)
        diameter_row.grid(row=row, column=0, sticky="ew")
        diameter_row.columnconfigure(0, weight=1)
        self.diameter_spinbox = ttk.Spinbox(
            diameter_row,
            from_=1,
            to=500,
            textvariable=self.brush_diameter,
            width=8,
            command=self._brush_diameter_changed,
        )
        self.diameter_spinbox.grid(row=0, column=0, sticky="ew")
        self.diameter_spinbox.bind("<FocusOut>", self._brush_diameter_changed)
        self.diameter_spinbox.bind("<Return>", self._brush_diameter_changed)
        row += 1
        ttk.Label(
            right,
            text="组内图片逐格随机，旋转角度也逐格从六个方向随机。",
            wraplength=250,
            justify=tk.LEFT,
        ).grid(row=row, column=0, sticky="w", pady=(3, 0))
        row += 1
        ttk.Label(right, textvariable=self.tier_label, wraplength=250, justify=tk.LEFT).grid(
            row=row, column=0, sticky="w", pady=(6, 0)
        )
        row += 1
        ttk.Checkbutton(
            right,
            text="显示右下角比例尺",
            variable=self.show_scale,
            command=self._toggle_scale,
        ).grid(row=row, column=0, sticky="w", pady=(5, 0))
        row += 1

        self.gpu_button = ttk.Button(
            right, text="打开/重启 GPU 主视口", command=self.open_gpu, state=tk.DISABLED
        )
        self.gpu_button.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        row += 1
        self.flat_button = ttk.Button(
            right, text="打开 2D 展开编辑器", command=self.open_flat_editor, state=tk.DISABLED
        )
        self.flat_button.grid(row=row, column=0, sticky="ew", pady=(5, 0))
        row += 1
        self.aggregate_button = ttk.Button(
            right, text="重建多层星球远景缓存", command=self.build_aggregate_cache, state=tk.DISABLED
        )
        self.aggregate_button.grid(row=row, column=0, sticky="ew", pady=(5, 0))
        row += 1
        self.save_button = ttk.Button(right, text="保存星球地图", command=self.save, state=tk.DISABLED)
        self.save_button.grid(row=row, column=0, sticky="ew", pady=(5, 0))
        row += 1
        if self.main_window:
            ttk.Button(
                right,
                text="运行最终验收与 Windows 诊断",
                command=lambda: AcceptanceViewer(self.window, self.paths),
            ).grid(row=row, column=0, sticky="ew", pady=(5, 0))
            row += 1

        ttk.Separator(right).grid(row=row, column=0, sticky="ew", pady=10)
        row += 1
        ttk.Label(right, textvariable=self.summary, wraplength=260, justify=tk.LEFT).grid(
            row=row, column=0, sticky="ew"
        )
        row += 1
        ttk.Label(
            right,
            text=(
                "球面视图和2D展开视图表示同一张完整地图。球面中右键拖动旋转，滚轮缩放。"
                "左侧选择分类文件夹作为随机笔刷组，按住左键连续拖画。"
                "每格随机抽取组内图片与六方向旋转，新笔划会完整覆盖旧状态。"
            ),
            wraplength=260,
            justify=tk.LEFT,
        ).grid(row=row, column=0, sticky="w", pady=(8, 0))

        status_bar = ttk.Frame(self.window, relief=tk.SUNKEN, padding=(8, 3))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.view_load_progress = ttk.Progressbar(
            status_bar,
            variable=self.view_load_value,
            maximum=100.0,
            length=150,
            mode="determinate",
        )
        self.view_load_progress.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Label(
            status_bar,
            textvariable=self.view_load_text,
            anchor=tk.E,
            width=31,
        ).pack(side=tk.RIGHT)
        ttk.Label(status_bar, textvariable=self.status, anchor=tk.W).pack(
            fill=tk.X, side=tk.LEFT, expand=True
        )

    def open_brush_cropper(self) -> None:
        BrushImageCropper(self.window, self.paths)

    def open_flat_editor(self) -> None:
        if self.topology is None or self.layout is None or self.session is None:
            self.status.set("请等待完整星球地图准备完成")
            return
        try:
            editor = ProductionFlatMapEditor(self)
        except Exception as exc:
            messagebox.showerror("2D展开编辑器", str(exc), parent=self.window)
            return
        self._flat_editors.add(editor)
        self.status.set("已打开2D二十面体展开编辑器；它与球面视图共用同一张地图")

    def refresh_brushes(self) -> None:
        if hasattr(self, "stroke_busy_count") and self._strokes_busy():
            self.status.set("请先松开鼠标并等待当前连续笔划完成，再刷新笔刷库")
            return
        if hasattr(self, "brush_prewarm_generation"):
            self.brush_prewarm_generation += 1
        try:
            self.scan_result = self.catalog.scan()
        except Exception as exc:
            messagebox.showerror("笔刷库", str(exc), parent=self.window)
            return
        self.records_by_uid = {record.uid: record for record in self.scan_result.records}
        self.brush_tree.delete(*self.brush_tree.get_children())
        categories: dict[str, str] = {}
        active = sorted(
            self.scan_result.active_records,
            key=lambda item: item.relative_path.casefold(),
        )
        for record in active:
            parent = ""
            path_parts: list[str] = []
            if record.category_path:
                for part in record.category_path.split("/"):
                    path_parts.append(part)
                    key = "/".join(path_parts)
                    if key not in categories:
                        categories[key] = self.brush_tree.insert(
                            parent, tk.END, text=part, open=False, tags=(f"group:{key}",)
                        )
                    parent = categories[key]
            else:
                key = ""
                if key not in categories:
                    categories[key] = self.brush_tree.insert(
                        "", tk.END, text="未分类", open=False, tags=("group:",)
                    )
                parent = categories[key]
            self.brush_tree.insert(
                parent,
                tk.END,
                text=Path(record.relative_path).name,
                tags=(f"uid:{record.uid}", f"group:{record.category_path}"),
            )
        retextured = False
        if self.controller is not None:
            retextured = self.controller.update_records(self.records_by_uid)
            # Brush pixels feed the far-view surface's representative colors, so
            # a content change has to rebuild it rather than reuse the signature.
            self._request_surface_build(force=retextured)
        if self.bridge is not None:
            self._stop_embedded_gpu()
            self.bridge.close()
            self.bridge = None
        if retextured:
            # The stream was cleared, so both viewports have to redraw from it.
            self._request_draw(0)
            for editor in tuple(self._flat_editors):
                editor.refresh_after_external_edit()
        if self.selected_brush_group is not None:
            selected_records = records_for_group(
                self.records_by_uid, self.selected_brush_group
            )
            if not selected_records:
                self.selected_brush_group = None
                self.selected_brush_uid = None
                self.selected_path.set("未选择笔刷组")
        self.status.set(
            f"笔刷扫描：有效 {self.scan_result.active_count}，缺失 {self.scan_result.missing_count}"
        )

    def _brush_selected(self, _event: tk.Event) -> None:
        selection = self.brush_tree.selection()
        if not selection:
            return
        tags = self.brush_tree.item(selection[0], "tags")
        group_tag = next((tag[6:] for tag in tags if tag.startswith("group:")), None)
        uid = next((tag[4:] for tag in tags if tag.startswith("uid:")), None)
        if group_tag is None and uid is not None:
            record = self.records_by_uid.get(uid)
            group_tag = None if record is None else record.category_path
        if group_tag is None:
            return
        records = records_for_group(self.records_by_uid, group_tag)
        if not records:
            self.status.set("该分类中没有有效的512×512 PNG笔刷")
            return
        self.selected_brush_group = group_tag
        self.selected_brush_uid = records[0].uid
        label = group_tag or "未分类"
        self.selected_path.set(f"笔刷组：{label}（{len(records)}张图片）")
        self.tool.set("paint")
        self._sync_tool()
        self._prewarm_brush_group(records)

    def _prewarm_brush_group(self, records: tuple[BrushRecord, ...]) -> None:
        self.brush_prewarm_generation += 1
        generation = self.brush_prewarm_generation
        level = self.tier_policy.tier.texture_lod
        if level is None or not records:
            return
        records = tuple(records)
        cache = BrushLodCache(self.paths.brush_root)
        self.status.set(
            f"正在后台预热笔刷组：{len(records)} 张 LOD{level} 纹理……"
        )

        def worker() -> None:
            generated = 0
            try:
                def ensure_current(record: BrushRecord):
                    if generation != self.brush_prewarm_generation:
                        return None
                    return cache.ensure(record, level)

                futures = [
                    self.parallel_executor.submit(ensure_current, record)
                    for record in records
                ]
                for future in as_completed(futures):
                    if generation != self.brush_prewarm_generation:
                        return
                    info = future.result()
                    if info is None:
                        continue
                    generated += int(info.generated)
                self.results.put(
                    (
                        "brush_prewarm_ready",
                        (generation, level, len(records), generated),
                    )
                )
            except Exception as exc:
                self.results.put(("brush_prewarm_error", (generation, exc)))

        self.background_executor.submit(worker)

    def _brush_diameter_value(self) -> int:
        try:
            value = int(self.brush_diameter.get())
        except (tk.TclError, ValueError):
            value = 1
        return self.tier_policy.clamp_diameter(max(1, min(500, value)))

    def _brush_diameter_changed(self, _event: tk.Event | None = None) -> None:
        value = self._brush_diameter_value()
        self.brush_diameter.set(value)
        self._sync_tool()
        self._refresh_tier_label()
        self._update_scale_bar()
        self._update_brush_cursor()

    def _sphere_geometry(self) -> tuple[float, float, float]:
        """Canvas center and projected planet radius, matching ``_draw_overview``."""
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        radius = min(width, height) * 0.42 * self.zoom
        return width / 2.0, height / 2.0, radius

    def _apply_zoom_tier(self) -> None:
        """Re-select the tier and snap the brush diameter into its range.

        The clamp is what makes a zoomed-out tier usable: at whole-planet scale a
        one-cell brush edits a real cell but changes nothing the viewer can see,
        and at maximum zoom a 500-cell brush would paint nine screens beyond the
        window in every direction.
        """
        previous = self.tier_policy.tier
        tier = self.tier_policy.update(self.zoom)
        raw = self.brush_diameter.get() if self.brush_diameter is not None else 1
        try:
            raw = int(raw)
        except (tk.TclError, ValueError):
            raw = 1
        clamped = tier.clamp_diameter(max(1, min(500, raw)))
        if clamped != raw:
            self.brush_diameter.set(clamped)
            self._sync_tool()
        if tier.index != previous.index:
            self.status.set(
                f"缩放档位切换为 {tier.name}（{tier.description}）；"
                f"笔刷直径范围 {tier.min_diameter}～{tier.max_diameter} 格"
            )
            if self.selected_brush_group is not None:
                records = records_for_group(
                    self.records_by_uid, self.selected_brush_group
                )
                self._prewarm_brush_group(records)
        self._refresh_tier_label()
        self._update_scale_bar()

    def _refresh_tier_label(self) -> None:
        tier = self.tier_policy.tier
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        cells = visible_cells_estimate(self.zoom, width, height)
        pixels = cell_pixels(self.zoom, width, height, frequency=self.FREQUENCY)
        self.tier_label.set(
            f"档位 {tier.name}：{tier.description}\n"
            f"缩放 {self.zoom:.2f}×｜可见约 {cells:,} 格｜{pixels:.2f} px/格\n"
            f"该档笔刷 {tier.min_diameter}～{tier.max_diameter} 格，"
            f"反馈粒度约 {tier.feedback_cells} 格"
        )

    def _toggle_scale(self) -> None:
        if self.show_scale.get():
            self.scale_frame.place(
                relx=1.0,
                rely=1.0,
                x=-16,
                y=-16,
                anchor="se",
            )
            self._update_scale_bar()
        else:
            self.scale_frame.place_forget()

    def _update_scale_bar(self) -> None:
        if not self.show_scale.get():
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        pixels_per_cell = cell_pixels(
            self.zoom,
            width,
            height,
            frequency=self.FREQUENCY,
        )
        pixels_per_km = pixels_per_cell / self.CELL_SIZE_KM
        target_pixels = 135.0
        raw_km = target_pixels / max(1e-9, pixels_per_km)
        exponent = math.floor(math.log10(max(1e-9, raw_km)))
        candidates = [
            factor * (10.0 ** power)
            for power in range(exponent - 2, exponent + 3)
            for factor in (1.0, 2.0, 5.0)
        ]
        usable = [
            distance
            for distance in candidates
            if 65.0 <= distance * pixels_per_km <= 165.0
        ]
        choices = usable or candidates
        distance_km = min(
            choices,
            key=lambda value: abs(value * pixels_per_km - target_pixels),
        )
        bar_pixels = max(8.0, min(165.0, distance_km * pixels_per_km))
        right = 174.0
        left = right - bar_pixels
        self.scale_canvas.coords(self.scale_bar_item, left, 8, right, 8)
        self.scale_canvas.coords(self.scale_left_tick_item, left, 3, left, 12)
        self.scale_canvas.coords(self.scale_right_tick_item, right, 3, right, 12)
        if distance_km >= 100.0:
            distance_label = f"{distance_km:,.0f}"
        elif distance_km >= 10.0:
            distance_label = f"{distance_km:.1f}".rstrip("0").rstrip(".")
        else:
            distance_label = f"{distance_km:.2f}".rstrip("0").rstrip(".")
        self.scale_text.set(f"{distance_label} km")
        self.scale_frame.lift()

    def _refresh_undo_label(self) -> None:
        history = self.undo_history
        self.undo_button.configure(
            state=tk.NORMAL if history.can_undo else tk.DISABLED
        )
        self.redo_button.configure(
            state=tk.NORMAL if history.can_redo else tk.DISABLED
        )
        if not history.can_undo and not history.can_redo:
            self.undo_label.set("撤销栈：空")
            return
        megabytes = history.byte_size / (1024.0 * 1024.0)
        self.undo_label.set(
            f"撤销栈：{history.undo_depth} 笔可撤销，"
            f"{history.redo_depth} 笔可重做（{megabytes:.1f} MB）"
        )

    def _canvas_configure(self, event: tk.Event) -> None:
        if self.mini_frame is not None:
            self.mini_frame.lift()
        self._update_scale_bar()
        viewport = self.embedded_gpu_viewport
        if viewport is not None:
            viewport.resize(event.width, event.height)
            return
        self._request_draw()

    def _stop_embedded_gpu(self, *, wait: bool = False) -> None:
        viewport = self.embedded_gpu_viewport
        self.embedded_gpu_viewport = None
        self.embedded_gpu_active = False
        self.embedded_gpu_starting = False
        if viewport is not None:
            viewport.close(wait=wait, timeout=1.0)

    def _maybe_start_embedded_gpu(self) -> None:
        if (
            not self.main_window
            or not native_gpu_supported()
            or self.controller is None
            or self.session is None
            or self.surface_texture is None
            or self.embedded_gpu_active
            or self.embedded_gpu_starting
            or self.bridge is not None
        ):
            return
        self.embedded_gpu_starting = True
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.status.set("正在启动嵌入式 GPU 主视口……")

        def worker() -> None:
            try:
                frame = self.controller.update_view(
                    self.yaw, self.pitch, self.zoom, width, height
                )
                batch = self.controller.initial_batch()
                self.results.put(("embedded_gpu_ready", (frame, batch, width, height)))
            except Exception as exc:
                self.results.put(("embedded_gpu_error", exc))

        self.background_executor.submit(worker)

    def _start_prepare(self) -> None:
        if self.worker_running:
            return
        self.worker_running = True
        self.status.set("正在生成或读取 f1004 生产布局与可见性索引……")
        self.background_executor.submit(self._prepare_worker)

    def _prepare_worker(self) -> None:
        try:
            cache = self.paths.map_root / ".topology" / "ico_dual_f1004_v2"
            if (cache / "production.json").exists():
                layout = load_production_layout_cache(cache)
            else:
                topology = ProductionTopology(self.FREQUENCY)
                layout = ProductionChunkLayout(topology, tile_side=self.TILE_SIDE)
                write_production_layout_cache(layout, self.paths.map_root / ".topology")
            if (cache / "production_visibility.json").exists():
                visibility = load_production_visibility_cache(cache, layout)
            else:
                visibility = build_production_visibility_index(layout)
                write_production_visibility_cache(visibility, self.paths.map_root / ".topology")
            self.results.put(("prepared", (layout, visibility)))
        except Exception as exc:
            self.results.put(("error", exc))

    def _start_single_map(self) -> None:
        if self.single_map_name is None or self.layout is None or self.worker_running:
            return
        self.worker_running = True
        self.status.set("正在创建或打开唯一的完整星球地图……")
        map_name = self.single_map_name

        def worker() -> None:
            try:
                if map_name in self.store.list_maps():
                    session = self.store.open(map_name, self.layout)
                else:
                    session = self.store.create_blank(map_name, self.layout)
                self.results.put(("opened", session))
            except Exception as exc:
                self.results.put(("error", exc))

        self.background_executor.submit(worker)

    def _refresh_maps(self) -> None:
        if self.single_map_name is not None or self.map_combo is None or self.open_button is None:
            return
        layout = self.layout
        if layout is None:
            return
        names: list[str] = []
        for name in self.store.list_maps():
            try:
                self.store.open(name, layout)
            except Exception:
                continue
            names.append(name)
        self.map_combo.configure(values=tuple(names))
        if names and self.map_choice.get() not in names:
            self.map_choice.set(names[0])
        self.open_button.configure(state=tk.NORMAL if names else tk.DISABLED)

    def create_map(self) -> None:
        if self.worker_running or self.layout is None:
            return
        name = self.map_name.get().strip()
        if not name:
            messagebox.showerror("地图", "地图名称不能为空", parent=self.window)
            return
        self.worker_running = True
        self.status.set("正在创建完整 f1004 Pack 地图……")

        def worker() -> None:
            try:
                session = self.store.create_blank(name, self.layout)
                self.results.put(("opened", session))
            except Exception as exc:
                self.results.put(("error", exc))

        self.background_executor.submit(worker)

    def open_map(self) -> None:
        if self.layout is None:
            return
        name = self.map_choice.get().strip()
        if not name:
            return
        try:
            session = self.store.open(name, self.layout)
        except Exception as exc:
            messagebox.showerror("打开地图", str(exc), parent=self.window)
            return
        self._set_session(session)

    def _set_session(self, session: SphereMapSession) -> None:
        for editor in tuple(self._flat_editors):
            try:
                editor.close()
            except Exception:
                pass
        self._flat_editors.clear()
        self._stop_embedded_gpu()
        self.mini_render_generation += 1
        self.mini_last_key = None
        self.pending_mini_render = None
        if self.mini_render_job is not None:
            try:
                self.window.after_cancel(self.mini_render_job)
            except tk.TclError:
                pass
            self.mini_render_job = None
        if self.mini_frame is not None:
            self.mini_canvas.itemconfigure(self.mini_image_item, image="")
            self.mini_canvas.itemconfigure(
                self.mini_message_item,
                text="正在载入宏观地表…",
                state=tk.NORMAL,
            )
        with self.surface_live_lock:
            self.surface_live_generation += 1
            self.surface_live_state = None
            self.surface_live_prepare_running = False
            self.surface_live_update_running = False
            self.surface_live_dirty_chunks.clear()
        # Snapshots reference the local brush ids of the session they were taken
        # from, so they cannot survive a session swap.
        self.undo_history.clear()
        self._refresh_undo_label()
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        assert self.topology is not None and self.layout is not None and self.visibility is not None
        if self.controller is not None:
            self.controller.close()
        self.session = session
        self.controller = ProductionGpuStreamingController(
            self.topology,
            self.layout,
            self.visibility,
            session,
            self.store,
            self.records_by_uid,
            self.paths.brush_root,
            lod_level=4,
            automatic_lod=True,
            stream_add_budget=192,
            stream_remove_budget=384,
            parallel_executor=self.parallel_executor,
            parallel_workers=self.PARALLEL_WORKERS,
        )
        self.map_name.set(session.name)
        self.save_button.configure(state=tk.NORMAL)
        self.gpu_button.configure(state=tk.NORMAL)
        self.aggregate_button.configure(state=tk.NORMAL)
        self.flat_button.configure(state=tk.NORMAL)
        self.status.set("完整星球地图已就绪" if self.single_map_name else f"已打开生产地图：{session.name}")
        self._request_surface_build()
        self._draw_overview()
        if self.auto_open_gpu and not self._auto_gpu_started and native_gpu_supported():
            self._auto_gpu_started = True
            self.window.after(250, self.open_gpu)

    def open_gpu(self) -> None:
        if self.controller is None or self.session is None:
            self.status.set("请先创建或打开生产地图")
            return
        if not native_gpu_supported():
            messagebox.showinfo(
                "GPU 编辑器",
                "原生 WGL/OpenGL 主视口只在 Windows 上启用；当前平台继续使用独立进程软件视口。",
                parent=self.window,
            )
            return
        if self.main_window:
            self._stop_embedded_gpu()
            if self.bridge is not None:
                self.bridge.close()
                self.bridge = None
            if self.surface_texture is None:
                self.status.set("缩略地表尚未准备完成，完成后会自动启动 GPU 主视口")
                self._request_surface_build()
                return
            self._maybe_start_embedded_gpu()
            return
        if self.worker_running:
            return
        self.worker_running = True
        self.gpu_button.configure(state=tk.DISABLED)
        self.status.set("正在读取初始可见区块并生成 GPU 实例流……")

        def worker() -> None:
            try:
                frame = self.controller.update_view(self.yaw, self.pitch, self.zoom, 1100, 760)
                batch = self.controller.initial_batch()
                self.results.put(("gpu_ready", (frame, batch)))
            except Exception as exc:
                self.results.put(("error", exc))

        self.background_executor.submit(worker)

    def _launch_gpu(self, frame, batch) -> None:
        self.bridge = GpuEditBridge(
            initial_state=self._tool_state(),
            known_texture_keys=(layer.key for layer in batch.texture_layers),
        )
        launch = launch_gpu_preview(
            batch,
            self.yaw,
            self.pitch,
            self.zoom,
            topology=self.topology,
            edit_bridge=self.bridge,
            title=(
                f"Hex Planet Production GPU Editor - {batch.instance_count:,} instances / "
                f"{len(frame.update.active_chunk_ids):,} chunks"
            ),
            streaming=True,
            min_zoom=0.8,
            editable=True,
            # Without this, a GPU window opened after the surface cache was built
            # would never receive the texture (the surface_ready push only reaches
            # the bridge that existed at build time) and its far view would be a
            # blank shaded sphere.
            surface_texture=self.surface_texture,
            maximum_texture_layers=self.controller.texture_layer_limit,
            texture_memory_budget_bytes=self.controller.texture_memory_budget_bytes,
            on_capabilities=lambda limit: self.results.put(
                ("gpu_capabilities", limit)
            ),
            on_error=lambda message: self.results.put(("gpu_error", message)),
        )
        if not launch.started:
            self.bridge.close()
            self.bridge = None
            messagebox.showerror("GPU 编辑器", launch.reason, parent=self.window)
            return
        self.status.set(
            f"生产 GPU 编辑器已启动：区块 {len(frame.update.active_chunk_ids):,}，实例 {batch.instance_count:,}"
        )

    def _launch_embedded_gpu(self, frame, batch, width: int, height: int) -> None:
        self.embedded_gpu_starting = False
        self.window.update_idletasks()
        self.bridge = GpuEditBridge(
            initial_state=self._tool_state(),
            known_texture_keys=(layer.key for layer in batch.texture_layers),
        )
        launch = launch_gpu_preview(
            batch,
            self.yaw,
            self.pitch,
            self.zoom,
            topology=self.topology,
            edit_bridge=self.bridge,
            title="Hex Planet Embedded GPU Viewport",
            streaming=True,
            min_zoom=0.8,
            # Editable at every LOD: the far view paints against the map through
            # the analytic pick, with incremental feedback on the live surface.
            editable=True,
            surface_texture=self.surface_texture,
            parent_hwnd=int(self.canvas.winfo_id()),
            initial_width=width,
            initial_height=height,
            maximum_texture_layers=self.controller.texture_layer_limit,
            texture_memory_budget_bytes=self.controller.texture_memory_budget_bytes,
            on_capabilities=lambda limit: self.results.put(
                ("gpu_capabilities", limit)
            ),
            on_error=lambda message: self.results.put(("embedded_gpu_error", message)),
        )
        if not launch.started or launch.viewport is None:
            self.bridge.close()
            self.bridge = None
            self.status.set(f"嵌入式 GPU 主视口启动失败，继续使用软件视口：{launch.reason}")
            self._request_draw(0)
            return
        self.embedded_gpu_viewport = launch.viewport
        self.embedded_gpu_active = True
        self._cancel_final_surface_render()
        self.surface_renderer.close()
        self.canvas.delete("detail")
        self.canvas.delete("fallback")
        self.canvas.itemconfigure(self.surface_item, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.surface_outline_item, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.overlay_item, state=tk.HIDDEN)
        self.gpu_button.configure(text="GPU 主视口已启用", state=tk.DISABLED)
        self.status.set(
            f"GPU 主视口已启用：区块 {len(frame.update.active_chunk_ids):,}，"
            f"实例 {batch.instance_count:,}"
        )
        if self.mini_frame is not None:
            self.mini_frame.lift()
            self.window.after(100, self.mini_frame.lift)
        if self.show_scale.get():
            self.scale_frame.lift()
            self.window.after(100, self.scale_frame.lift)
        self.window.after(80, lambda: launch.viewport.resize(
            max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        ))

    def _tool_state(self) -> GpuToolState:
        records = (
            ()
            if self.selected_brush_group is None
            else records_for_group(self.records_by_uid, self.selected_brush_group)
        )
        record = records[0] if records else None
        return GpuToolState(
            tool=self.tool.get(),
            brush_uid=None if record is None else record.uid,
            last_known_path="" if record is None else record.relative_path,
            rotation=0,
            brush_group=self.selected_brush_group or "",
            brush_diameter=self._brush_diameter_value(),
        )

    def _sync_tool(self) -> None:
        if self.bridge is not None and not self.bridge.closed:
            try:
                self.bridge.set_tool_state(self._tool_state())
            except ValueError as exc:
                self.status.set(str(exc))

    def _mark_view_loading(self) -> None:
        self.view_load_progress.stop()
        self.view_load_progress.configure(mode="indeterminate")
        self.view_load_progress.start(12)
        self.view_load_text.set("视图：正在计算可见区块…")

    def _update_view_loading(self, patch: GpuStreamPatch | GpuBatchResetPatch) -> None:
        self.view_load_progress.stop()
        self.view_load_progress.configure(mode="determinate")
        loaded = max(0, int(patch.loaded_chunk_count))
        total = max(0, int(patch.total_chunk_count))
        remaining = max(0, int(patch.remaining_chunk_count))
        if total <= 0:
            self.view_load_value.set(100.0)
            self.view_load_text.set("视图：远景地表已就绪")
            return
        percent = max(0, min(100, round(loaded * 100 / total)))
        self.view_load_value.set(float(percent))
        if patch.has_more or remaining > 0 or loaded < total:
            suffix = f"，待处理 {remaining:,}" if remaining else ""
            self.view_load_text.set(
                f"视图：加载 {loaded:,}/{total:,}（{percent}%）{suffix}"
            )
        else:
            self.view_load_text.set(
                f"视图：已就绪 {loaded:,}/{total:,}（100%）"
            )

    def _poll_bridge(self) -> None:
        bridge = self.bridge
        if bridge is None:
            return
        if bridge.closed:
            self.bridge = None
            self.view_load_progress.stop()
            self.view_load_progress.configure(mode="determinate")
            self.view_load_value.set(0.0)
            self.view_load_text.set("视图：GPU 已关闭")
            if self.embedded_gpu_active:
                self.embedded_gpu_active = False
                self.embedded_gpu_viewport = None
                self.gpu_button.configure(text="打开/重启 GPU 主视口", state=tk.NORMAL)
                self.status.set("GPU 主视口已关闭，已回退到软件视口")
                if self.surface_texture is not None:
                    try:
                        self.surface_renderer.set_texture(self.surface_texture)
                    except Exception as exc:
                        self.status.set(f"GPU 主视口已关闭；软件后备视口启动失败：{exc}")
                self._request_draw(0)
            return
        bridge_state = bridge.tool_state()
        self.tool.set(bridge_state.tool)
        self.rotation.set(bridge_state.rotation)
        self.brush_diameter.set(bridge_state.brush_diameter)
        latest_view: GpuViewRequest | None = None
        for request in bridge.poll_requests():
            if isinstance(request, GpuViewRequest):
                latest_view = request
            elif isinstance(request, GpuEditRequest):
                self._handle_edit(request)
            elif isinstance(request, GpuSaveRequest):
                self._handle_save(request)
            elif isinstance(request, GpuResyncRequest):
                self._start_gpu_resync(request, bridge)
        if latest_view is not None:
            self.yaw = latest_view.yaw
            self.pitch = latest_view.pitch
            self.zoom = latest_view.zoom
            self._apply_zoom_tier()
            self.pending_view = latest_view
            self._mark_view_loading()
            self._request_mini_globe()
        self._start_view_worker()

    def _start_view_worker(self) -> None:
        if (
            self.view_worker_running
            or self.pending_view is None
            or self.controller is None
            or self.save_running
            or self.gpu_resync_running
        ):
            return
        request = self.pending_view
        self.pending_view = None
        self.last_view_request = request
        self.view_worker_running = True
        controller = self.controller
        bridge = self.bridge

        def worker() -> None:
            try:
                logging.debug(
                    "GPU view worker start request=%s zoom=%.3f size=%sx%s interactive=%s",
                    request.request_id,
                    request.zoom,
                    request.width,
                    request.height,
                    request.interactive,
                )
                # Queue the patch while still holding the same lock that mutates
                # the controller stream. This preserves the exact mutation order
                # against concurrent brush edits and prevents deltas from
                # overtaking an LOD full reset.
                with controller.lock:
                    pending = self.pending_view
                    if (
                        pending is not None
                        and pending.request_id > request.request_id
                    ):
                        self.results.put(
                            ("view_superseded", request.request_id)
                        )
                        return
                    frame = controller.update_view(
                        request.yaw,
                        request.pitch,
                        request.zoom,
                        request.width,
                        request.height,
                        interactive=request.interactive,
                    )
                    pending = self.pending_view
                    if (
                        pending is not None
                        and pending.request_id > request.request_id
                    ):
                        # The producer changed, but this camera is already stale.
                        # Do not make the GPU apply a soon-to-be-undone delta.
                        # The next stable view sends one complete snapshot.
                        controller.request_consumer_reset()
                        self.results.put(
                            ("view_superseded", request.request_id)
                        )
                        return
                    logging.debug(
                        "GPU view worker updated request=%s lod=%s instances=%s",
                        request.request_id,
                        frame.lod.level,
                        frame.update.instance_count,
                    )
                    patch = controller.patch_for_frame(request.request_id, frame)
                    self.results.put(("view_patch", (bridge, patch)))
            except Exception as exc:
                self.results.put(("view_error", (request.request_id, exc)))

        self.background_executor.submit(worker)

    def _start_gpu_resync(
        self,
        request: GpuResyncRequest,
        bridge: GpuEditBridge,
    ) -> None:
        controller = self.controller
        if (
            controller is None
            or self.gpu_resync_running
            or bridge is not self.bridge
            or bridge.closed
        ):
            return
        self.gpu_resync_running = True
        self.status.set("检测到 GPU 数据流不同步，正在自动重建完整视图……")
        logging.warning(
            "GPU full resync requested id=%s reason=%s",
            request.request_id,
            request.reason,
        )

        def worker() -> None:
            try:
                with controller.lock:
                    batch = controller.stream.snapshot_batch(
                        controller.topology,
                        controller.layout,
                    )
                    patch = GpuBatchResetPatch(
                        request_id=request.request_id,
                        batch=batch,
                        message=(
                            f"GPU 数据流已自动恢复：{batch.instance_count:,} 个实例"
                        ),
                        editable=True,
                    )
                    self.results.put(("gpu_resync_ready", (bridge, patch)))
            except Exception as exc:
                self.results.put(("gpu_resync_error", (bridge, exc)))

        self.background_executor.submit(worker)

    def _increment_stroke_busy(self) -> None:
        with self.stroke_busy_lock:
            self.stroke_busy_count += 1

    def _decrement_stroke_busy(self) -> int:
        with self.stroke_busy_lock:
            self.stroke_busy_count = max(0, self.stroke_busy_count - 1)
            return self.stroke_busy_count

    def _strokes_busy(self) -> bool:
        with self.stroke_busy_lock:
            return self.stroke_busy_count > 0

    def _queue_stroke_segment(
        self,
        source: str,
        stroke_id: int,
        phase: str,
        cell_id: int,
        tool: BrushStrokeTool,
        request_id: int = 0,
    ) -> None:
        command: tuple[object, ...] = (
            source, int(stroke_id), phase, int(cell_id), tool, int(request_id)
        )
        if phase == "move":
            with self.stroke_move_lock:
                first = int(stroke_id) not in self.stroke_pending_moves
                self.stroke_pending_moves[int(stroke_id)] = command
            if first:
                self._increment_stroke_busy()
                self.stroke_command_queue.put(("move_marker", int(stroke_id)))
            return
        self._increment_stroke_busy()
        self.stroke_command_queue.put(command)

    def _stroke_worker_loop(self) -> None:
        while not self.stroke_worker_stop.is_set():
            try:
                command = self.stroke_command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command and command[0] == "stop":
                return
            if command and command[0] == "move_marker":
                stroke_id = int(command[1])
                with self.stroke_move_lock:
                    actual = self.stroke_pending_moves.pop(stroke_id, None)
                if actual is None:
                    if self._decrement_stroke_busy() == 0:
                        self.results.put(("stroke_idle", None))
                    continue
                command = actual
            source, stroke_id, phase, cell_id, tool, request_id = command
            try:
                touched, changed = self._apply_stroke_segment(
                    int(stroke_id),
                    str(phase),
                    int(cell_id),
                    tool,
                    request_id=int(request_id),
                )
                self.results.put((
                    "stroke_result",
                    (str(source), int(stroke_id), str(phase), touched, changed, tool),
                ))
            except Exception as exc:
                self.results.put((
                    "stroke_error",
                    (str(source), int(stroke_id), str(phase), int(request_id), exc),
                ))
            finally:
                self.stroke_command_queue.task_done()
                if self._decrement_stroke_busy() == 0:
                    self.results.put(("stroke_idle", None))

    def _stroke_tool_from_state(self, state: GpuToolState) -> BrushStrokeTool:
        if state.tool in {"erase", "undo"}:
            return BrushStrokeTool(state.tool, state.brush_diameter).validated()
        if state.brush_uid is None:
            raise SphereMapError("请先在左侧选择一个笔刷分类文件夹")
        records = records_for_group(self.records_by_uid, state.brush_group)
        if not records and state.brush_uid is not None:
            record = self.records_by_uid.get(state.brush_uid)
            if record is not None and record.state == "active":
                records = (record,)
        return BrushStrokeTool("paint", state.brush_diameter, records).validated()

    def _stroke_tool_from_ui(self) -> BrushStrokeTool:
        diameter = self._brush_diameter_value()
        if self.tool.get() in {"erase", "undo"}:
            return BrushStrokeTool(self.tool.get(), diameter).validated()
        if self.selected_brush_group is None:
            raise SphereMapError("请先在左侧选择一个笔刷分类文件夹")
        records = records_for_group(self.records_by_uid, self.selected_brush_group)
        return BrushStrokeTool("paint", diameter, records).validated()

    @staticmethod
    def _undo_label(tool: BrushStrokeTool) -> str:
        action = {"paint": "绘制", "erase": "清除", "undo": "撤销笔刷"}.get(
            tool.tool, tool.tool
        )
        return f"{action} 直径{tool.diameter}格"

    def _undo_brush_values(self, session: SphereMapSession, cell_ids) -> dict[int, int]:
        """Previous value for each cell under the undo brush.

        Cells with no recorded history are left untouched rather than cleared, so
        brushing over never-painted terrain does nothing instead of erasing it.
        """
        history = self.undo_history
        layout = session.layout
        values: dict[int, int] = {}
        for cell_id in cell_ids:
            if cell_id < 12:
                continue
            chunk_id, local_index = layout.chunk_for_cell(int(cell_id))
            previous = history.previous_value(chunk_id, local_index)
            if previous is not None:
                values[int(cell_id)] = previous
        return values

    def undo(self) -> None:
        self._undo_or_redo(redo=False)

    def redo(self) -> None:
        self._undo_or_redo(redo=True)

    def _undo_or_redo(self, *, redo: bool) -> None:
        session = self.session
        controller = self.controller
        layout = self.layout
        name = "重做" if redo else "撤销"
        if session is None or controller is None or layout is None:
            return
        if self._strokes_busy():
            self.status.set(f"正在等待连续笔划完成，请稍后再{name}")
            return
        bridge = self.bridge
        try:
            with controller.lock:
                result = (
                    self.undo_history.redo(session, self.store, layout)
                    if redo
                    else self.undo_history.undo(session, self.store, layout)
                )
                if result is None:
                    self.status.set(f"没有可{name}的笔划")
                    self._refresh_undo_label()
                    return
                patches, uploads, released_layers = controller.patch_visible_cells(
                    result.changed_cell_ids
                )
                if (
                    bridge is not None
                    and not bridge.closed
                    and (patches or uploads or released_layers)
                ):
                    self.results.put((
                        "gpu_patch",
                        (
                            bridge,
                            GpuStreamPatch(
                                request_id=0,
                                active_chunk_ids=tuple(
                                    sorted(controller.stream.active_chunks)
                                ),
                                instance_count=controller.stream.instance_count,
                                removed_cell_ids=(),
                                added=(),
                                changed=tuple(patches),
                                texture_uploads=tuple(uploads),
                                message=(
                                    f"{name}：{len(result.changed_cell_ids):,} 格"
                                ),
                                lod_level=controller.lod_level,
                                padded_size=controller.stream.lod_cache.padded_size(
                                    controller.lod_level
                                ),
                                released_texture_layers=tuple(released_layers),
                                editable=True,
                            ),
                        ),
                    ))
        except (UndoHistoryError, SphereMapError) as exc:
            messagebox.showerror(name, str(exc), parent=self.window)
            return
        self.status.set(
            f"{name}「{result.label}」：{len(result.changed_cell_ids):,} 格，"
            f"{result.chunk_count} 个区块；远景纹理正在局部更新"
        )
        self._queue_surface_live_cells(result.changed_cell_ids)
        self._refresh_undo_label()
        self._request_draw(0)
        for editor in tuple(self._flat_editors):
            editor.refresh_after_external_edit()

    def _apply_stroke_segment(
        self,
        stroke_id: int,
        phase: str,
        cell_id: int,
        tool: BrushStrokeTool,
        *,
        request_id: int = 0,
    ) -> tuple[int, int]:
        session = self.session
        planner = self.stroke_planner
        controller = self.controller
        if session is None or planner is None or controller is None:
            raise SphereMapError("完整星球地图尚未准备完成")
        if phase == "end":
            state = self.active_strokes.pop(int(stroke_id), None)
            with controller.lock:
                self.undo_history.commit(0 if state is None else state.touched_cell_count)
            return (0 if state is None else state.touched_cell_count, 0)
        if phase not in {"point", "start", "move"}:
            raise SphereMapError(f"不支持的笔划阶段：{phase}")
        if cell_id < 0 or cell_id >= planner.topology.cell_count:
            raise SphereMapError(f"CellId超出范围：{cell_id}")

        key = int(stroke_id)
        state = self.active_strokes.get(key)
        if phase in {"point", "start"} or state is None:
            state = BrushStrokeState(tool=tool.validated())
            self.active_strokes[key] = state
            self.undo_history.begin(self._undo_label(state.tool))
        plan = planner.plan_segment(state, int(cell_id))
        bridge = self.bridge
        with controller.lock:
            journal = self.undo_history.capture if self.undo_history.capturing else None
            if state.tool.tool == "undo":
                changed_cell_ids = session.set_raw_values(
                    self.store,
                    self._undo_brush_values(session, plan.affected_cell_ids),
                    journal=journal,
                )
            elif state.tool.tool == "erase":
                changed_cell_ids = session.clear_cells(
                    self.store,
                    plan.affected_cell_ids,
                    journal=journal,
                )
            else:
                changed_cell_ids = session.paint_cells_random(
                    self.store,
                    plan.affected_cell_ids,
                    state.tool.records,
                    self.stroke_rng,
                    journal=journal,
                )
            patches, uploads, released_layers = controller.patch_visible_cells(
                changed_cell_ids
            )
            if bridge is not None and not bridge.closed:
                if patches or uploads or released_layers:
                    gpu_patch = GpuStreamPatch(
                        request_id=request_id,
                        active_chunk_ids=tuple(sorted(controller.stream.active_chunks)),
                        instance_count=controller.stream.instance_count,
                        removed_cell_ids=(),
                        added=(),
                        changed=tuple(patches),
                        texture_uploads=tuple(uploads),
                        message=(
                            f"连续笔划更新 {len(changed_cell_ids):,} 格；"
                            f"直径 {state.tool.diameter} 格"
                        ),
                        lod_level=controller.lod_level,
                        padded_size=controller.stream.lod_cache.padded_size(
                            controller.lod_level
                        ),
                        released_texture_layers=tuple(released_layers),
                        editable=True,
                    )
                else:
                    # ``controller.lod_level`` mirrors the instance stream, which
                    # is capped at 3; the far-view decision lives on the LOD
                    # controller.
                    far_view = controller.lod_controller.level >= 4
                    if changed_cell_ids and far_view:
                        note = (
                            f"远景笔划已写入 {len(changed_cell_ids):,} 格；"
                            "正在局部更新 L5 地表纹理"
                        )
                    else:
                        note = (
                            f"笔划经过 {len(plan.affected_cell_ids):,} 格；"
                            "当前格子不在GPU实例流或状态未变化"
                        )
                    gpu_patch = GpuStatusPatch(request_id, True, note)
                # This queue insertion happens before releasing controller.lock,
                # so its order matches every view/reset mutation exactly.
                self.results.put(("gpu_patch", (bridge, gpu_patch)))

        self._queue_surface_live_cells(changed_cell_ids)
        if phase == "point":
            self.active_strokes.pop(key, None)
        return len(plan.affected_cell_ids), len(changed_cell_ids)

    def _handle_edit(self, request: GpuEditRequest) -> None:
        bridge = self.bridge
        if bridge is None:
            return
        try:
            if request.phase == "end":
                self._queue_stroke_segment(
                    "gpu",
                    request.stroke_id,
                    "end",
                    request.cell_id,
                    BrushStrokeTool("erase", 1),
                    request.request_id,
                )
                return
            tool = self._stroke_tool_from_state(request.state)
            self._queue_stroke_segment(
                "gpu",
                request.stroke_id or request.request_id,
                request.phase,
                request.cell_id,
                tool,
                request.request_id,
            )
        except Exception as exc:
            bridge.push_patch(GpuStatusPatch(request.request_id, False, f"编辑失败：{exc}"))

    def _handle_save(self, request: GpuSaveRequest) -> None:
        bridge = self.bridge
        if bridge is None or self.controller is None:
            return
        if self._strokes_busy():
            self.pending_gpu_saves.append(request)
            bridge.push_patch(
                GpuStatusPatch(request.request_id, True, "正在等待连续笔划处理完成后保存")
            )
            return
        self._execute_gpu_save(request)

    def _execute_gpu_save(self, request: GpuSaveRequest) -> None:
        bridge = self.bridge
        if bridge is None or self.controller is None:
            return
        if self.save_running:
            self.pending_gpu_saves.append(request)
            bridge.push_patch(
                GpuStatusPatch(request.request_id, True, "已有后台保存任务，已排队")
            )
            return
        self._start_save_worker(request)

    def build_aggregate_cache(self) -> None:
        if self.controller is None or self.worker_running:
            return
        self.worker_running = True
        self.aggregate_button.configure(state=tk.DISABLED)
        self.status.set("正在流式构建多层聚合远景缓存……")

        def worker() -> None:
            try:
                report = self.controller.build_aggregate_cache()
                self.results.put(("aggregate_ready", report))
            except Exception as exc:
                self.results.put(("error", exc))

        self.background_executor.submit(worker)

    def save(self) -> None:
        if self.controller is None:
            return
        if self._strokes_busy():
            self.pending_local_save = True
            self.status.set("正在等待连续笔划处理完成，随后自动保存")
            return
        self._execute_local_save()

    def _execute_local_save(self) -> None:
        if self.controller is None:
            return
        if self.save_running:
            self.pending_local_save = True
            self.status.set("已有后台保存任务，本次保存已排队")
            return
        self._start_save_worker(None)

    def _start_save_worker(self, request: GpuSaveRequest | None) -> None:
        controller = self.controller
        if controller is None or self.save_running:
            return
        dirty = len(controller.session.dirty_chunks)
        if dirty == 0 and not controller.session.brush_table_dirty:
            message = "当前没有未保存修改"
            self.status.set(message)
            if request is not None and self.bridge is not None and not self.bridge.closed:
                self.bridge.push_patch(
                    GpuStatusPatch(request.request_id, True, message)
                )
            return
        self.save_running = True
        self.save_started_at = time.monotonic()
        self.save_stage = "waiting"
        self.save_status_second = -1
        self.save_button.configure(state=tk.DISABLED)
        self.aggregate_button.configure(state=tk.DISABLED)
        message = (
            f"保存已提交：等待当前视图计算结束（{dirty:,} 个脏区块）……"
        )
        self.status.set(message)
        logging.info("Save queued dirty_chunks=%s", dirty)
        if request is not None and self.bridge is not None and not self.bridge.closed:
            self.bridge.push_patch(GpuStatusPatch(request.request_id, True, message))

        def worker() -> None:
            try:
                def progress(stage: str, completed: int, total: int) -> None:
                    self.results.put((
                        "save_progress",
                        (request, stage, int(completed), int(total), dirty),
                    ))

                saved = controller.save(progress=progress)
                self.results.put(("save_ready", (request, saved)))
            except Exception as exc:
                self.results.put(("save_error", (request, exc)))

        worker_thread = threading.Thread(
            target=worker,
            name="hexplanet-pack-save",
            daemon=True,
        )
        self.save_worker = worker_thread
        worker_thread.start()

    def _resume_pending_save(self) -> None:
        if self.save_running or self._strokes_busy():
            return
        if self.pending_local_save:
            self.pending_local_save = False
            self._execute_local_save()
            return
        if self.pending_gpu_saves:
            request = self.pending_gpu_saves.pop(0)
            self._execute_gpu_save(request)

    def _queue_surface_live_cells(self, cell_ids) -> None:
        layout = self.layout
        if layout is None:
            return
        chunk_ids = {
            layout.chunk_for_cell(int(cell_id))[0]
            for cell_id in cell_ids
            if 0 <= int(cell_id) < layout.cell_count
        }
        if not chunk_ids:
            return
        with self.surface_live_lock:
            self.surface_live_dirty_chunks.update(chunk_ids)
        self._start_surface_live_update()

    def _install_surface_live_image(self, image) -> None:
        with self.surface_live_lock:
            state = self.surface_live_state
        if state is not None:
            state.replace_image(image)
            session = self.session
            if session is not None:
                with self.surface_live_lock:
                    self.surface_live_dirty_chunks.update(session.dirty_chunks)
            self._start_surface_live_update()
            return
        self._start_surface_live_prepare(image)

    def _start_surface_live_prepare(self, image) -> None:
        visibility = self.visibility
        if visibility is None:
            return
        with self.surface_live_lock:
            self.surface_live_generation += 1
            generation = self.surface_live_generation
            self.surface_live_prepare_running = True
            session = self.session
            if session is not None:
                self.surface_live_dirty_chunks.update(session.dirty_chunks)

        def worker() -> None:
            try:
                state = ProductionSurfaceLiveState(visibility, image)
                self.results.put(("surface_live_prepared", (generation, state)))
            except Exception as exc:
                self.results.put(("surface_live_error", (generation, exc)))

        self.background_executor.submit(worker)

    def _start_surface_live_update(self) -> None:
        controller = self.controller
        session = self.session
        if controller is None or session is None:
            return
        with self.surface_live_lock:
            state = self.surface_live_state
            if (
                state is None
                or self.surface_live_update_running
                or not self.surface_live_dirty_chunks
            ):
                return
            chunk_ids = tuple(sorted(self.surface_live_dirty_chunks))
            self.surface_live_dirty_chunks.clear()
            self.surface_live_update_running = True
            generation = self.surface_live_generation
        records = dict(self.records_by_uid)

        def worker() -> None:
            try:
                with controller.lock:
                    local_colors = self.surface_cache.local_colors(session, records)
                    chunk_values = {
                        chunk_id: tuple(
                            session.loaded_chunks.get(chunk_id)
                            or self.store.read_chunk_values(session, chunk_id)
                        )
                        for chunk_id in chunk_ids
                    }
                region, image = state.update_chunks(chunk_values, local_colors)
                self.results.put(
                    (
                        "surface_live_ready",
                        (generation, region, image, len(chunk_ids)),
                    )
                )
            except Exception as exc:
                self.results.put(("surface_live_update_error", (generation, exc)))

        self.background_executor.submit(worker)

    def _request_surface_build(self, *, force: bool = False) -> None:
        session = self.session
        layout = self.layout
        visibility = self.visibility
        if session is None or layout is None or visibility is None:
            return
        if self.surface_build_running:
            self.surface_rebuild_requested = self.surface_rebuild_requested or force
            return
        self.surface_build_running = True
        self.surface_rebuild_requested = False
        snapshot = SphereMapSession(
            name=session.name,
            map_dir=session.map_dir,
            layout=layout,
            brush_entries=dict(session.brush_entries),
            index_records=list(session.index_records),
            chunks_per_pack=session.chunks_per_pack,
        )
        records = dict(self.records_by_uid)
        self.status.set("正在生成缩小后的星球地表缓存……")

        def worker() -> None:
            try:
                info, image = self.surface_cache.ensure(
                    layout, visibility, snapshot, self.store, records
                )
                self.results.put(("surface_ready", (info, image)))
            except Exception as exc:
                self.results.put(("surface_error", exc))

        self.background_executor.submit(worker)

    def _update_mini_reticle(self) -> None:
        frame = self.mini_frame
        if frame is None:
            return
        center = self.MINI_GLOBE_DIAMETER / 2.0 + 4.0
        sphere_radius = self.MINI_GLOBE_DIAMETER / 2.0 - 3.0
        zoom = max(0.8, float(self.zoom))
        viewport_ratio = min(
            1.0,
            0.5 / (VIEWPORT_RADIUS_FACTOR * zoom),
        )
        radius = max(3.0, sphere_radius * viewport_ratio)
        self.mini_canvas.coords(
            self.mini_reticle_item,
            center - radius,
            center - radius,
            center + radius,
            center + radius,
        )

    def _request_mini_globe(self, delay_ms: int = 35) -> None:
        if self.mini_frame is None:
            return
        self._update_mini_reticle()
        texture = self.surface_texture
        if texture is None:
            return
        key = (
            id(texture),
            round(self.yaw, 4),
            round(self.pitch, 4),
        )
        if key == self.mini_last_key and self.pending_mini_render is None:
            return

        def queue_latest() -> None:
            self.mini_render_job = None
            current_texture = self.surface_texture
            if current_texture is None:
                return
            current_key = (
                id(current_texture),
                round(self.yaw, 4),
                round(self.pitch, 4),
            )
            if current_key == self.mini_last_key:
                return
            self.mini_render_generation += 1
            generation = self.mini_render_generation
            self.pending_mini_render = (
                generation,
                current_key,
                current_texture,
                float(self.yaw),
                float(self.pitch),
            )
            self._start_mini_render()

        if self.mini_render_job is not None:
            try:
                self.window.after_cancel(self.mini_render_job)
            except tk.TclError:
                pass
        self.mini_render_job = self.window.after(max(0, int(delay_ms)), queue_latest)

    def _start_mini_render(self) -> None:
        if self.mini_render_running or self.pending_mini_render is None:
            return
        generation, key, texture, yaw, pitch = self.pending_mini_render
        self.pending_mini_render = None
        self.mini_render_running = True

        def worker() -> None:
            try:
                ppm = render_textured_sphere_ppm(
                    texture,
                    yaw,
                    pitch,
                    self.MINI_GLOBE_DIAMETER,
                    block_size=2,
                )
                self.results.put(("mini_globe_rendered", (generation, key, ppm)))
            except Exception as exc:
                self.results.put(("mini_globe_error", (generation, exc)))

        self.background_executor.submit(worker)

    def _queue_surface_render(
        self, width: int, height: int, center_x: float, center_y: float, radius: float
    ) -> None:
        texture = self.surface_texture
        if texture is None:
            return
        preview = self.dragging or self.surface_preview_mode
        preview_scale = 4 if preview else 1
        render_width = max(1, (int(width) + preview_scale - 1) // preview_scale)
        render_height = max(1, (int(height) + preview_scale - 1) // preview_scale)
        render_center_x = center_x / preview_scale
        render_center_y = center_y / preview_scale
        render_radius = radius / preview_scale
        block_size = 4 if preview else (3 if max(width, height) >= 900 else 2)
        key = (
            self.surface_signature,
            round(self.yaw, 5),
            round(self.pitch, 5),
            int(width),
            int(height),
            round(float(radius), 2),
            preview_scale,
            block_size,
        )
        self.surface_latest_key = key
        if key == self.surface_photo_key:
            return
        self.surface_render_generation += 1
        generation = self.surface_render_generation
        self.pending_surface_render = (
            generation, key, texture, self.yaw, self.pitch, render_width, render_height,
            float(render_center_x), float(render_center_y), float(render_radius),
            block_size, preview_scale,
        )
        self._start_surface_render()

    def _start_surface_render(self) -> None:
        if self.pending_surface_render is None:
            return
        request = self.pending_surface_render
        generation, key, texture, yaw, pitch, width, height, center_x, center_y, radius, block_size, preview_scale = request
        if self.surface_renderer.running:
            self.pending_surface_render = None
            self.surface_render_running = True
            try:
                self.surface_renderer.submit(
                    SoftwareRenderRequest(
                        generation=generation,
                        key=key,
                        yaw=yaw,
                        pitch=pitch,
                        width=width,
                        height=height,
                        center_x=center_x,
                        center_y=center_y,
                        radius=radius,
                        block_size=block_size,
                        preview_scale=preview_scale,
                    )
                )
                return
            except Exception as exc:
                self.status.set(f"独立渲染进程不可用，已回退到线程：{exc}")
        if self.surface_render_running:
            return
        self.pending_surface_render = None
        self.surface_render_running = True

        def worker() -> None:
            try:
                ppm = render_textured_globe_view_ppm(
                    texture, yaw, pitch, width, height, center_x, center_y, radius,
                    block_size=block_size,
                )
                self.results.put((
                    "surface_rendered",
                    (generation, key, ppm, preview_scale),
                ))
            except Exception as exc:
                self.results.put(("surface_render_error", exc))

        self.background_executor.submit(worker)

    def _draw_surface_base(
        self, width: int, height: int, center_x: float, center_y: float, radius: float
    ) -> None:
        self._queue_surface_render(width, height, center_x, center_y, radius)
        self.canvas.delete("fallback")
        if self.surface_photo is not None:
            self.canvas.itemconfigure(self.surface_item, image=self.surface_photo, state=tk.NORMAL)
            self.canvas.coords(self.surface_item, 0, 0)
            self.canvas.tag_lower(self.surface_item)
            return
        self.canvas.itemconfigure(self.surface_item, state=tk.HIDDEN)
        if radius > max(width, height) * 1.8:
            self.canvas.create_rectangle(
                0, 0, width, height, fill="#334751", outline="", tags=("fallback",)
            )
        else:
            draw_shaded_sphere(self.canvas, center_x, center_y, radius, tags=("fallback",))

    def _request_draw(self, delay_ms: int = 16) -> None:
        if self.embedded_gpu_active:
            return
        delay_ms = max(0, int(delay_ms))
        due_at = time.monotonic() + delay_ms / 1000.0
        if self._draw_job is not None:
            if due_at >= self._draw_due_at:
                return
            try:
                self.window.after_cancel(self._draw_job)
            except tk.TclError:
                pass
            self._draw_job = None
        self._draw_due_at = due_at

        def draw() -> None:
            self._draw_job = None
            self._draw_due_at = 0.0
            if self.window.winfo_exists():
                self._draw_overview()

        self._draw_job = self.window.after(delay_ms, draw)

    def _cancel_final_surface_render(self) -> None:
        if self._final_surface_job is None:
            return
        try:
            self.window.after_cancel(self._final_surface_job)
        except tk.TclError:
            pass
        self._final_surface_job = None

    def _schedule_final_surface_render(self, delay_ms: int = 100) -> None:
        self._cancel_final_surface_render()

        def finalize() -> None:
            self._final_surface_job = None
            self.surface_preview_mode = False
            self._request_draw(0)

        self._final_surface_job = self.window.after(max(0, int(delay_ms)), finalize)

    def _set_outline(self, center_x: float, center_y: float, radius: float, visible: bool) -> None:
        if not visible:
            self.canvas.itemconfigure(self.surface_outline_item, state=tk.HIDDEN)
            return
        self.canvas.coords(
            self.surface_outline_item,
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        self.canvas.itemconfigure(self.surface_outline_item, state=tk.NORMAL)
        self.canvas.tag_raise(self.surface_outline_item)

    def _set_overlay(self, text: str) -> None:
        self.canvas.itemconfigure(self.overlay_item, text=text, state=tk.NORMAL)
        self.canvas.tag_raise(self.overlay_item)

    def _draw_overview(self) -> None:
        if self.embedded_gpu_active:
            return
        self.canvas.delete("detail")
        self.canvas.delete("fallback")
        self.canvas.itemconfigure(self.center_message_item, state=tk.HIDDEN)
        self.canvas.itemconfigure(self.overlay_item, state=tk.HIDDEN)
        self.visible_cell_polygons = ()
        if self.embedded_gpu_active:
            self.canvas.itemconfigure(self.surface_item, state=tk.HIDDEN)
            self.canvas.itemconfigure(self.surface_outline_item, state=tk.HIDDEN)
            return
        visibility = self.visibility
        layout = self.layout
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        base_radius = min(width, height) * 0.42
        radius = base_radius * self.zoom
        cx, cy = width / 2.0, height / 2.0

        self._draw_surface_base(width, height, cx, cy, radius)

        if visibility is None or layout is None:
            self.canvas.coords(self.center_message_item, cx, cy)
            self.canvas.itemconfigure(
                self.center_message_item,
                text="正在准备完整星球……",
                state=tk.NORMAL,
            )
            self.canvas.tag_raise(self.center_message_item)
            self._set_outline(cx, cy, radius, radius <= max(width, height) * 2.0)
            return

        if self.dragging or self.surface_preview_mode:
            self._set_overlay(
                "唯一地图：整个星球\n"
                "显示模式：低延迟缩略地表预览\n"
                f"缩放 {self.zoom:.2f}×（停止操作后恢复精细显示）"
            )
            self._set_outline(cx, cy, radius, radius <= max(width, height) * 2.0)
            return

        query = visibility.query(layout, self.yaw, self.pitch, self.zoom, width, height)
        mode_text = (
            "缩略地表球面" if self.surface_texture is not None
            else "正在生成缩略地表缓存"
        )
        if self.session is not None and self.session.dirty_chunks and self.surface_texture is not None:
            mode_text += "（包含未保存的实时修改）"
        displayed = 0

        # The old gate was a hard ``zoom >= 36``, which sat in the middle of a LOD
        # band and stopped per-cell rendering well before the cell budget was
        # actually reached.  The budget is the real constraint, so the tier only
        # says whether per-cell drawing is meaningful at all.
        detail_requested = (
            self.tier_policy.tier.render in {"cell_texture", "cell_color"}
            and query.candidate_cells <= self.detail_cell_limit
        )
        if detail_requested and self.session is not None and self.topology is not None:
            active_chunks = set(query.chunk_ids)
            for chunk_id in query.chunk_ids:
                self.store.load_chunk(self.session, int(chunk_id))
            polygons = build_projected_cells(
                self.topology,
                layout,
                self.store,
                self.session,
                self.records_by_uid,
                self.paths.brush_root,
                query.chunk_ids,
                self.yaw,
                self.pitch,
                cx,
                cy,
                radius,
                width,
                height,
                maximum_cells=self.detail_cell_limit,
            )
            for chunk_id in tuple(self.session.loaded_chunks):
                if chunk_id not in active_chunks and chunk_id not in self.session.dirty_chunks:
                    self.session.loaded_chunks.pop(chunk_id, None)
            if polygons:
                self.visible_cell_polygons = polygons
                for polygon in polygons:
                    self.canvas.create_polygon(
                        polygon.points,
                        fill=polygon.fill,
                        outline=polygon.fill,
                        width=2,
                        tags=("detail",),
                    )
                if average_edge_pixels(polygons[len(polygons) // 2].points) >= 4.0:
                    for polygon in polygons:
                        self.canvas.create_line(
                            *polygon.points,
                            polygon.points[0],
                            polygon.points[1],
                            fill=polygon.edge,
                            width=1,
                            joinstyle=tk.ROUND,
                            tags=("detail",),
                        )
                displayed = len(polygons)
                mode_text = "真实共享边界六边形（按住左键连续绘制）"
            else:
                mode_text = "已保存地表（仍可绘制，放大后才逐格显示）"
        elif self.tier_policy.tier.render in {"cell_texture", "cell_color"}:
            mode_text = "候选格子超出逐格预算，放大后才逐格显示（仍可绘制）"

        tier = self.tier_policy.tier
        self._set_outline(cx, cy, radius, radius <= max(width, height) * 2.0)
        self._set_overlay(
            "唯一地图：整个星球\n"
            f"档位 {tier.name}：{tier.description}（本档可绘制）\n"
            f"显示模式：{mode_text}\n"
            f"当前详细格子：{displayed:,}；候选格子：{query.candidate_cells:,}\n"
            f"缩放 {self.zoom:.2f}×｜笔刷 {self._brush_diameter_value()} 格"
            f"（{tier.min_diameter}～{tier.max_diameter}）"
        )
        self.canvas.tag_raise("cursor")

    def _hit_canvas_cell(self, x: int, y: int) -> ProjectedCellPolygon | None:
        for polygon in reversed(self.visible_cell_polygons):
            if point_in_polygon(x, y, polygon.points):
                return polygon
        return None

    def _pick_cell(self, x: float, y: float) -> int | None:
        """Resolve a canvas position to a CellId at any zoom level.

        Detail polygons are consulted first only because they are already there;
        the analytic picker is what removes the old requirement to be zoomed past
        36x before anything could be painted.  Its precision is one cell at every
        tier, so what a far tier loses is feedback resolution, not edit accuracy.
        """
        hit = self._hit_canvas_cell(int(x), int(y))
        if hit is not None:
            return hit.cell_id
        picker = self.picker
        if picker is None:
            return None
        center_x, center_y, radius = self._sphere_geometry()
        try:
            return picker.pick(x, y, center_x, center_y, radius, self.yaw, self.pitch)
        except Exception:
            return None

    def _paint_start(self, event: tk.Event) -> None:
        if self.session is None or self.picker is None:
            self.status.set("请等待完整星球地图准备完成")
            return
        cell_id = self._pick_cell(event.x, event.y)
        if cell_id is None:
            self.status.set("点击位置在星球轮廓之外")
            return
        try:
            tool = self._stroke_tool_from_ui()
        except Exception as exc:
            messagebox.showerror("球面编辑", str(exc), parent=self.window)
            return
        if tool.tool == "undo" and not self.undo_history.can_undo:
            self.status.set("撤销栈为空：撤销笔刷只能回退本次会话中已记录的笔划")
            return
        self.local_stroke_id += 1
        stroke_id = -self.local_stroke_id
        self.local_stroke_active = True
        self.local_stroke_tool = tool
        self._queue_stroke_segment(
            "local", stroke_id, "start", cell_id, tool
        )
        action = {"paint": "随机覆盖", "erase": "清除", "undo": "撤销"}.get(
            tool.tool, tool.tool
        )
        self.status.set(
            f"开始连续{action}，直径 {tool.diameter} 格，档位 "
            f"{self.tier_policy.tier.name}；正在计算首个笔刷区域"
        )

    def _paint_move(self, event: tk.Event) -> None:
        self._track_cursor(event)
        if not self.local_stroke_active or self.local_stroke_tool is None:
            return
        cell_id = self._pick_cell(event.x, event.y)
        if cell_id is None:
            return
        self._queue_stroke_segment(
            "local",
            -self.local_stroke_id,
            "move",
            cell_id,
            self.local_stroke_tool,
        )

    def _track_cursor(self, event: tk.Event) -> None:
        self.cursor_position = (int(event.x), int(event.y))
        self._update_brush_cursor()

    def _clear_cursor(self, _event: tk.Event | None = None) -> None:
        self.cursor_position = None
        self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)

    def _update_brush_cursor(self) -> None:
        """Draw the brush footprint where it will actually land.

        The footprint is dilated over the cell graph, so its screen size follows
        the planet, not the window.  Showing it is the only way to tell, at a
        zoomed-out tier, whether a stroke covers a province or a continent.
        """
        position = self.cursor_position
        if position is None or self.embedded_gpu_active or self.picker is None:
            self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)
            return
        center_x, center_y, radius = self._sphere_geometry()
        direction = screen_to_direction(
            position[0], position[1], center_x, center_y, radius, self.yaw, self.pitch
        )
        if direction is None:
            self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)
            return
        graph_radius = max(0, self._brush_diameter_value() // 2)
        angular = (graph_radius + 0.5) * cell_angular_pitch(self.FREQUENCY)
        points: list[float] = []
        for sample in cap_boundary_directions(direction, angular, samples=48):
            screen_x, screen_y, depth = project_direction(
                sample, center_x, center_y, radius, self.yaw, self.pitch
            )
            # Samples behind the limb have no projection; dropping them leaves the
            # visible arc rather than folding the outline through the planet.
            if depth <= 0.0:
                continue
            points.extend((screen_x, screen_y))
        if len(points) < 6:
            self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)
            return
        outline = {"paint": "#ffd166", "erase": "#ff8f8f", "undo": "#8fd0ff"}.get(
            self.tool.get(), "#ffd166"
        )
        self.canvas.coords(self.brush_cursor_item, *points)
        self.canvas.itemconfigure(
            self.brush_cursor_item, state=tk.NORMAL, outline=outline
        )
        self.canvas.tag_raise(self.brush_cursor_item)

    def _paint_end(self, _event: tk.Event | None = None) -> None:
        if not self.local_stroke_active:
            return
        self.local_stroke_active = False
        stroke_id = -self.local_stroke_id
        tool = self.local_stroke_tool or BrushStrokeTool("erase", 1)
        self.local_stroke_tool = None
        self._queue_stroke_segment("local", stroke_id, "end", -1, tool)
        self.status.set("鼠标已松开，正在完成连续笔划并合并脏区块")

    def _rotate(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        return rotate_point(point, self.yaw, self.pitch)

    def _drag_start(self, event: tk.Event) -> None:
        self.drag_origin = (
            event.x,
            event.y,
            self.yaw,
            self.pitch,
            drag_radians_per_pixel(self.zoom),
        )
        self.dragging = True
        self.surface_preview_mode = True
        self._cancel_final_surface_render()

    def _drag_move(self, event: tk.Event) -> None:
        if self.drag_origin is None:
            return
        x, y, yaw, pitch, radians_per_pixel = self.drag_origin
        self.yaw = yaw + (event.x - x) * radians_per_pixel
        self.pitch = max(
            -1.45,
            min(1.45, pitch + (event.y - y) * radians_per_pixel),
        )
        self._request_mini_globe()
        self._request_draw(16)

    def _drag_end(self, _event: tk.Event) -> None:
        self.drag_origin = None
        self.dragging = False
        self.surface_preview_mode = True
        self._request_mini_globe(0)
        self._request_draw(0)
        self._schedule_final_surface_render(100)

    def _wheel(self, event: tk.Event) -> None:
        self._zoom(event.delta / 120.0)

    def _zoom(self, steps: float) -> None:
        self.zoom = max(0.8, min(512.0, self.zoom * (1.16 ** steps)))
        self._apply_zoom_tier()
        self._request_mini_globe(0)
        self.surface_preview_mode = True
        self._cancel_final_surface_render()
        self._request_draw(0)
        self._schedule_final_surface_render(100)
        self._update_brush_cursor()

    def _poll(self) -> None:
        self._poll_bridge()
        render_result = self.surface_renderer.poll_latest()
        if render_result is not None:
            if render_result.error is None:
                self.results.put((
                    "surface_rendered",
                    (
                        render_result.generation,
                        render_result.key,
                        render_result.ppm,
                        render_result.preview_scale,
                    ),
                ))
            else:
                self.results.put(("surface_render_error", render_result.error))
        try:
            result_deadline = time.perf_counter() + 0.008
            processed_results = 0
            while processed_results < 128 and time.perf_counter() < result_deadline:
                kind, payload = self.results.get_nowait()
                processed_results += 1
                if kind == "prepared":
                    layout, visibility = payload
                    self.worker_running = False
                    self.layout = layout
                    self.topology = layout.topology
                    self.stroke_planner = BrushStrokePlanner(self.topology)
                    self.picker = SphereScreenPicker(self.topology)
                    self.visibility = visibility
                    self._apply_zoom_tier()
                    self.create_button.configure(state=tk.NORMAL) if self.create_button is not None else None
                    self.summary.set(
                        f"CellId：{layout.cell_count:,}\n"
                        f"普通六边形：{layout.topology.hexagon_count:,}\n"
                        f"逻辑区块：{layout.chunk_count:,}\n"
                        f"可见性节点：{visibility.node_count:,}\n"
                        "展开的全局单格记录：0"
                    )
                    self.status.set("完整星球拓扑与可见性索引已就绪")
                    self._refresh_maps()
                    self._draw_overview()
                    if self.single_map_name is not None:
                        self.window.after(10, self._start_single_map)
                elif kind == "opened":
                    self.worker_running = False
                    self._set_session(payload)
                    self._refresh_maps()
                elif kind == "aggregate_ready":
                    self.worker_running = False
                    self.aggregate_button.configure(state=tk.NORMAL)
                    self.status.set(
                        f"聚合远景缓存完成：节点 {payload.node_count:,}，"
                        f"新生成 {payload.generated_count:,}，复用 {payload.reused_count:,}"
                    )
                elif kind == "surface_ready":
                    self.surface_build_running = False
                    info, image = payload
                    self.surface_texture = image
                    self.surface_signature = info.signature
                    self.surface_photo = None
                    self.surface_photo_key = None
                    self.surface_latest_key = None
                    self.surface_render_generation += 1
                    self.mini_last_key = None
                    try:
                        self.surface_renderer.set_texture(image)
                        renderer_note = "独立渲染进程已启动"
                    except Exception as exc:
                        renderer_note = f"独立渲染进程启动失败，使用线程后备：{exc}"
                    self.status.set(
                        ("缩略地表缓存已生成" if info.generated else "缩略地表缓存已载入")
                        + f"；{renderer_note}"
                    )
                    if self.bridge is not None and not self.bridge.closed:
                        self.bridge.push_patch(
                            GpuSurfaceTexturePatch(
                                request_id=0,
                                width=image.width,
                                height=image.height,
                                channels=image.channels,
                                pixels=image.pixels,
                            )
                        )
                    self._request_draw(0)
                    self._maybe_start_embedded_gpu()
                    self._install_surface_live_image(image)
                    self._request_mini_globe(0)
                    if self.surface_rebuild_requested:
                        self._request_surface_build(force=True)
                elif kind == "surface_error":
                    self.surface_build_running = False
                    self.status.set(f"缩略地表缓存失败：{payload}")
                    if self.surface_rebuild_requested:
                        self._request_surface_build(force=True)
                elif kind == "surface_live_prepared":
                    generation, state = payload
                    with self.surface_live_lock:
                        if generation != self.surface_live_generation:
                            continue
                        self.surface_live_state = state
                        self.surface_live_prepare_running = False
                    self._start_surface_live_update()
                elif kind == "surface_live_ready":
                    generation, region, image, chunk_count = payload
                    with self.surface_live_lock:
                        if generation != self.surface_live_generation:
                            continue
                        self.surface_live_update_running = False
                    self.surface_texture = image
                    self._request_mini_globe()
                    if region is not None:
                        if self.bridge is not None and not self.bridge.closed:
                            self.bridge.push_patch(
                                GpuSurfaceRegionPatch(
                                    request_id=0,
                                    x=region.x,
                                    y=region.y,
                                    width=region.width,
                                    height=region.height,
                                    channels=3,
                                    pixels=region.pixels_rgb,
                                    message=(
                                        f"L5 远景实时更新：{chunk_count:,} 个区块，"
                                        f"{region.width}×{region.height} 纹理区域"
                                    ),
                                )
                            )
                        if not self.embedded_gpu_active:
                            try:
                                self.surface_renderer.set_texture(image)
                            except Exception as exc:
                                self.status.set(f"L5 实时纹理更新失败：{exc}")
                            self._request_draw(0)
                        for editor in tuple(self._flat_editors):
                            editor.refresh_after_external_edit()
                    self._start_surface_live_update()
                elif kind == "surface_live_error":
                    generation, exc = payload
                    with self.surface_live_lock:
                        if generation != self.surface_live_generation:
                            continue
                        self.surface_live_prepare_running = False
                    self.status.set(f"L5 实时纹理索引失败：{exc}")
                elif kind == "surface_live_update_error":
                    generation, exc = payload
                    with self.surface_live_lock:
                        if generation != self.surface_live_generation:
                            continue
                        self.surface_live_update_running = False
                    self.status.set(f"L5 实时纹理更新失败：{exc}")
                    self._start_surface_live_update()
                elif kind == "surface_rendered":
                    self.surface_render_running = False
                    generation, key, ppm, preview_scale = payload
                    if (
                        generation == self.surface_render_generation
                        and key == self.surface_latest_key
                    ):
                        try:
                            photo = tk.PhotoImage(data=ppm, format="PPM")
                            if preview_scale > 1:
                                photo = photo.zoom(preview_scale, preview_scale)
                            self.surface_photo = photo
                            self.surface_photo_key = key
                            self.surface_displayed_generation = generation
                        except Exception as exc:
                            self.status.set(f"缩略地表显示失败：{exc}")
                    self._start_surface_render()
                    self._request_draw(0)
                elif kind == "surface_render_error":
                    self.surface_render_running = False
                    self.status.set(f"缩略地表渲染失败：{payload}")
                    self._start_surface_render()
                elif kind == "mini_globe_rendered":
                    self.mini_render_running = False
                    generation, key, ppm = payload
                    if (
                        self.mini_frame is not None
                        and generation == self.mini_render_generation
                    ):
                        try:
                            self.mini_photo = tk.PhotoImage(
                                master=self.mini_canvas,
                                data=ppm,
                                format="PPM",
                            )
                            self.mini_canvas.itemconfigure(
                                self.mini_image_item,
                                image=self.mini_photo,
                            )
                            self.mini_canvas.itemconfigure(
                                self.mini_message_item,
                                state=tk.HIDDEN,
                            )
                            self.mini_last_key = key
                            self.mini_frame.lift()
                        except Exception as exc:
                            self.status.set(f"宏观预览显示失败：{exc}")
                    self._start_mini_render()
                elif kind == "mini_globe_error":
                    self.mini_render_running = False
                    generation, exc = payload
                    if generation == self.mini_render_generation:
                        self.status.set(f"宏观预览渲染失败：{exc}")
                        if self.mini_frame is not None:
                            self.mini_canvas.itemconfigure(
                                self.mini_message_item,
                                text="宏观预览暂不可用",
                                state=tk.NORMAL,
                            )
                    self._start_mini_render()
                elif kind == "brush_prewarm_ready":
                    generation, level, count, generated = payload
                    if generation == self.brush_prewarm_generation:
                        self.status.set(
                            f"笔刷纹理预热完成：LOD{level} 共 {count} 张，"
                            f"新生成 {generated} 张"
                        )
                elif kind == "brush_prewarm_error":
                    generation, exc = payload
                    if generation == self.brush_prewarm_generation:
                        self.status.set(f"笔刷纹理预热失败：{exc}")
                elif kind == "save_progress":
                    request, stage, completed, total, dirty = payload
                    self.save_stage = stage
                    elapsed = max(0.0, time.monotonic() - self.save_started_at)
                    if stage == "prepare":
                        message = (
                            f"正在整理 {dirty:,} 个脏区块，准备写盘"
                            f"（已用 {elapsed:.1f} 秒）……"
                        )
                    elif stage == "brush_table":
                        message = "正在写入笔刷索引……"
                    elif stage == "pack":
                        message = (
                            f"正在写入地图 Pack：{completed:,}/{total:,}"
                            f"（已用 {elapsed:.1f} 秒）"
                        )
                    elif stage == "index":
                        message = "Pack 已写入，正在原子提交 index.bin……"
                    elif stage == "complete":
                        message = "磁盘写入完成，正在收尾……"
                    else:
                        message = f"正在保存：{stage}"
                    self.status.set(message)
                    if (
                        request is not None
                        and self.bridge is not None
                        and not self.bridge.closed
                    ):
                        self.bridge.push_patch(
                            GpuStatusPatch(request.request_id, True, message)
                        )
                elif kind == "save_ready":
                    request, dirty = payload
                    elapsed = max(0.0, time.monotonic() - self.save_started_at)
                    logging.info(
                        "Save completed dirty_chunks=%s elapsed=%.3fs",
                        dirty,
                        elapsed,
                    )
                    self.save_running = False
                    self.save_worker = None
                    self.save_stage = ""
                    self.save_status_second = -1
                    self.save_button.configure(
                        state=tk.NORMAL if self.session is not None else tk.DISABLED
                    )
                    self.aggregate_button.configure(
                        state=tk.NORMAL if self.session is not None else tk.DISABLED
                    )
                    message = f"保存完成：写入 {dirty} 个脏区块"
                    self.status.set(message)
                    if self.bridge is not None and not self.bridge.closed:
                        request_id = 0 if request is None else request.request_id
                        self.bridge.push_patch(GpuStatusPatch(request_id, True, message))
                    if dirty:
                        self._request_surface_build(force=True)
                    self._resume_pending_save()
                    self._start_view_worker()
                    if (
                        self.close_after_save
                        and not self.save_running
                        and not self.pending_local_save
                        and not self.pending_gpu_saves
                    ):
                        self.close_after_save = False
                        self.window.after(0, self._close)
                elif kind == "save_error":
                    request, exc = payload
                    logging.error(
                        "Save failed after %.3fs",
                        max(0.0, time.monotonic() - self.save_started_at),
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    self.save_running = False
                    self.save_worker = None
                    self.save_stage = ""
                    self.save_status_second = -1
                    self.close_after_save = False
                    self.save_button.configure(
                        state=tk.NORMAL if self.session is not None else tk.DISABLED
                    )
                    self.aggregate_button.configure(
                        state=tk.NORMAL if self.session is not None else tk.DISABLED
                    )
                    self.status.set(f"保存失败：{exc}")
                    if request is not None:
                        if self.bridge is not None and not self.bridge.closed:
                            self.bridge.push_patch(
                                GpuStatusPatch(request.request_id, False, f"保存失败：{exc}")
                            )
                    else:
                        messagebox.showerror("保存", str(exc), parent=self.window)
                    self._resume_pending_save()
                    self._start_view_worker()
                elif kind == "embedded_gpu_ready":
                    frame, batch, width, height = payload
                    self._launch_embedded_gpu(frame, batch, width, height)
                elif kind == "embedded_gpu_error":
                    self.embedded_gpu_starting = False
                    self.embedded_gpu_active = False
                    self.embedded_gpu_viewport = None
                    if self.bridge is not None:
                        self.bridge.close()
                        self.bridge = None
                    self.gpu_button.configure(text="打开/重启 GPU 主视口", state=tk.NORMAL)
                    self.status.set(f"嵌入式 GPU 主视口失败，已回退到软件视口：{payload}")
                    self._request_draw(0)
                elif kind == "gpu_ready":
                    self.worker_running = False
                    self.gpu_button.configure(state=tk.NORMAL)
                    frame, batch = payload
                    self._launch_gpu(frame, batch)
                elif kind == "stroke_result":
                    source, stroke_id, phase, touched, changed, tool = payload
                    action = {"paint": "随机覆盖", "erase": "清除", "undo": "撤销"}.get(
                        tool.tool, tool.tool
                    )
                    if phase == "end":
                        self._refresh_undo_label()
                        self.status.set(
                            f"连续笔划结束：本笔覆盖 {touched:,} 个去重格子；"
                            "Ctrl+Z 可整笔撤销，Ctrl+S 保存"
                        )
                    else:
                        self.status.set(
                            f"连续{action}：本段触及 {touched:,} 格，实际改写 {changed:,} 格，"
                            f"直径 {tool.diameter} 格"
                        )
                    self._request_draw(16 if phase != "end" else 0)
                elif kind == "stroke_error":
                    source, stroke_id, phase, request_id, exc = payload
                    self.active_strokes.pop(int(stroke_id), None)
                    if source == "gpu" and self.bridge is not None:
                        self.bridge.push_patch(
                            GpuStatusPatch(int(request_id), False, f"连续绘制失败：{exc}")
                        )
                    self.status.set(f"连续绘制失败：{exc}")
                elif kind == "stroke_idle":
                    if not self._strokes_busy():
                        self._resume_pending_save()
                elif kind == "gpu_patch":
                    target_bridge, patch = payload
                    if (
                        target_bridge is self.bridge
                        and not target_bridge.closed
                    ):
                        target_bridge.push_patch(patch)
                elif kind == "view_patch":
                    self.view_worker_running = False
                    target_bridge, patch = payload
                    self._update_view_loading(patch)
                    newer_pending = (
                        self.pending_view is not None
                        and self.pending_view.request_id > patch.request_id
                    )
                    # Controller patches are a serial state delta.  Even when a
                    # newer camera request is waiting, this patch must reach the
                    # GPU because the next delta is computed from its result.
                    if (
                        target_bridge is self.bridge
                        and not target_bridge.closed
                    ):
                        logging.debug(
                            "GPU queueing %s request=%s instances=%s newer_pending=%s",
                            type(patch).__name__,
                            getattr(patch, "request_id", None),
                            (
                                getattr(patch, "instance_count", None)
                                if not isinstance(patch, GpuBatchResetPatch)
                                else patch.batch.instance_count
                            ),
                            newer_pending,
                        )
                        target_bridge.push_patch(patch)
                    if newer_pending:
                        self._mark_view_loading()
                    self._start_view_worker()
                elif kind == "view_superseded":
                    self.view_worker_running = False
                    self._start_view_worker()
                elif kind == "gpu_resync_ready":
                    self.gpu_resync_running = False
                    target_bridge, patch = payload
                    if (
                        target_bridge is self.bridge
                        and not target_bridge.closed
                    ):
                        target_bridge.push_patch(patch)
                        self.status.set(patch.message)
                    self._start_view_worker()
                elif kind == "gpu_resync_error":
                    self.gpu_resync_running = False
                    target_bridge, exc = payload
                    logging.exception(
                        "GPU full resync failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    if target_bridge is self.bridge:
                        target_bridge.close()
                    self.status.set(f"GPU 自动恢复失败，已切换软件视图：{exc}")
                elif kind == "gpu_capabilities":
                    if self.controller is not None:
                        try:
                            limit = self.controller.request_hardware_texture_layer_limit(
                                int(payload)
                            )
                            self.status.set(
                                f"GPU 纹理数组层上限：{limit:,}；"
                                "显存缓存将在此范围内按 LRU 复用"
                            )
                        except Exception as exc:
                            self.status.set(f"GPU 纹理能力同步失败：{exc}")
                elif kind == "view_error":
                    self.view_worker_running = False
                    request_id, exc = payload
                    self.view_load_progress.stop()
                    self.view_load_progress.configure(mode="determinate")
                    self.view_load_value.set(0.0)
                    self.view_load_text.set("视图：加载失败")
                    if self.bridge is not None:
                        self.bridge.push_patch(GpuStatusPatch(request_id, False, str(exc)))
                    if self.pending_view is not None:
                        self._mark_view_loading()
                    self._start_view_worker()
                elif kind == "gpu_error":
                    self.status.set(f"GPU 窗口错误：{payload}")
                    messagebox.showerror("GPU 编辑器", str(payload), parent=self.window)
                elif kind == "error":
                    self.worker_running = False
                    self.gpu_button.configure(state=tk.NORMAL if self.session is not None else tk.DISABLED)
                    self.aggregate_button.configure(state=tk.NORMAL if self.session is not None else tk.DISABLED)
                    self.flat_button.configure(state=tk.NORMAL if self.session is not None else tk.DISABLED)
                    self.status.set(f"操作失败：{payload}")
                    messagebox.showerror("生产编辑器", str(payload), parent=self.window)
        except queue.Empty:
            pass
        if self.save_running and self.save_stage == "waiting":
            elapsed_second = max(0, int(time.monotonic() - self.save_started_at))
            if elapsed_second != self.save_status_second:
                self.save_status_second = elapsed_second
                self.status.set(
                    "保存已提交，正在等待当前视图计算释放地图锁"
                    f"（{elapsed_second} 秒）……"
                )
        if (
            self.save_running
            and self.save_worker is not None
            and not self.save_worker.is_alive()
            and self.results.empty()
        ):
            # A normal worker always posts save_ready/save_error, both of which
            # are drained above. Reaching here means it terminated abnormally;
            # never leave the UI permanently stuck in "saving".
            logging.error("Save worker terminated without a completion result")
            self.save_running = False
            self.save_worker = None
            self.save_stage = ""
            self.save_status_second = -1
            self.save_button.configure(
                state=tk.NORMAL if self.session is not None else tk.DISABLED
            )
            self.aggregate_button.configure(
                state=tk.NORMAL if self.session is not None else tk.DISABLED
            )
            self.status.set("保存线程异常结束，保存状态已解除；请重试并查看日志")
            self._start_view_worker()
        if self.window.winfo_exists():
            if not self.results.empty():
                delay = 1
            elif (
                self.embedded_gpu_active
                or self.view_worker_running
                or self.pending_view is not None
            ):
                delay = 8
            else:
                delay = 25
            self.window.after(delay, self._poll)

    def _close(self) -> None:
        if self.save_running:
            self.close_after_save = True
            self.status.set("地图正在后台保存；保存完成后将自动关闭")
            return
        session = self.session
        if (
            session is not None
            and (session.dirty_chunks or session.brush_table_dirty)
        ):
            unsaved_parts = [
                f"{len(session.dirty_chunks):,} 个地图区块"
                if session.dirty_chunks
                else ""
            ]
            if session.brush_table_dirty:
                unsaved_parts.append("笔刷索引")
            unsaved = "、".join(part for part in unsaved_parts if part)
            decision = messagebox.askyesnocancel(
                "关闭程序",
                (
                    f"还有未保存内容：{unsaved}。\n\n"
                    "是否保存后关闭？"
                ),
                parent=self.window,
            )
            if decision is None:
                return
            if decision:
                self.close_after_save = True
                self.save()
                return
        for editor in tuple(self._flat_editors):
            try:
                editor.close()
            except Exception:
                pass
        self._flat_editors.clear()
        self._stop_embedded_gpu(wait=True)
        if self.bridge is not None:
            self.bridge.close()
        self.stroke_worker_stop.set()
        self.stroke_command_queue.put(("stop",))
        self._cancel_final_surface_render()
        if self.mini_render_job is not None:
            try:
                self.window.after_cancel(self.mini_render_job)
            except tk.TclError:
                pass
            self.mini_render_job = None
        self.surface_renderer.close()
        if self.controller is not None:
            self.controller.close()
        self.parallel_executor.shutdown(wait=False, cancel_futures=True)
        self.background_executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()
