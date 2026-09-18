from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .async_loading import AsyncViewportChunkLoader
from .brush_catalog import BrushCatalog, BrushRecord, ScanResult
from .brush_lod import AsyncBrushLodBuilder, BrushLodCache, BrushLodPolicy, LOD_PADDING
from .chunk_layout import ChunkLayout, build_chunk_layout, write_chunk_layout_cache
from .chunk_visibility import (
    ChunkVisibilityIndex,
    build_chunk_visibility_index,
    write_chunk_visibility_cache,
)
from .distant_lod import (
    AsyncDistantLodBuilder,
    AsyncPlanetOverviewBuilder,
    DistantLodCache,
    polygon_center,
)
from .gpu_batch import (
    EMPTY_LAYER_KEY,
    MISSING_LAYER_KEY,
    GpuRenderBatchBuilder,
    brush_texture_key,
    build_brush_texture_payload,
)
from .gpu_edit import (
    GpuCellPatch,
    GpuEditBridge,
    GpuEditRequest,
    GpuSaveRequest,
    GpuStatusPatch,
    GpuToolState,
)
from .gpu_native import launch_gpu_preview, native_gpu_supported
from .paths import ProjectPaths
from .sphere_map_store import SphereMapError, SphereMapSession, SphereMapStore
from .sphere_viewport import ProjectedCell, SphereViewport, ViewportProjection
from .thumbnail import BrushThumbnailCache
from .topology import DualTopology, generate_dual_topology, write_topology_cache


class SphereMapEditor:
    def __init__(self, parent: tk.Misc, paths: ProjectPaths) -> None:
        self.paths = paths
        self.window = tk.Toplevel(parent)
        self.window.title("球面六边形地图编辑器 v1.3.1")
        self.window.geometry("1360x860")
        self.window.minsize(1040, 680)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.bind("<Control-s>", lambda _event: self.save_map())
        self.window.bind("<Destroy>", self._on_destroy, add="+")

        self.catalog = BrushCatalog(paths.brush_root)
        self.store = SphereMapStore(paths.map_root)
        self.chunk_loader = AsyncViewportChunkLoader(self.store, max_workers=2, max_inflight=8)
        self.lod_cache = BrushLodCache(paths.brush_root)
        self.lod_builder = AsyncBrushLodBuilder(self.lod_cache, max_workers=2, max_inflight=8)
        self.distant_cache = DistantLodCache(paths.brush_root)
        self.distant_builder = AsyncDistantLodBuilder(
            self.distant_cache, self.store, max_workers=2, max_inflight=8
        )
        self.planet_builder = AsyncPlanetOverviewBuilder(self.distant_cache, self.store)
        self.gpu_batch_builder = GpuRenderBatchBuilder(paths.brush_root)
        self.gpu_preview_pending = False
        self.gpu_preview_results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.gpu_edit_bridge: GpuEditBridge | None = None
        self.gpu_edit_lod_level: int | None = None
        self.lod_policy = BrushLodPolicy(initial_level=0)
        self.current_lod_level = 0
        self.scan_result: ScanResult | None = None
        self.records_by_uid: dict[str, BrushRecord] = {}
        self.selected_brush_uid: str | None = None
        self.topology: DualTopology | None = None
        self.layout: ChunkLayout | None = None
        self.visibility_index: ChunkVisibilityIndex | None = None
        self.session: SphereMapSession | None = None
        self.viewport = SphereViewport()
        self.projection: ViewportProjection | None = None
        self.drag_origin: tuple[int, int, float, float] | None = None
        self.selected_cell_id: int | None = None
        self.generating = False
        self.generation_results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.render_images: list[tk.PhotoImage] = []
        self.render_thumbnail_caches: dict[int, BrushThumbnailCache] = {}
        self.distant_photo_cache: dict[tuple[str, int], tk.PhotoImage] = {}
        self.overview_window: tk.Toplevel | None = None
        self.overview_label: ttk.Label | None = None
        self.overview_image: tk.PhotoImage | None = None
        self.overview_requested = False
        self.planet_refresh_requested = False
        self.preview_cache = BrushThumbnailCache(self.window, paths.brush_root, side=24)
        self.last_viewport_error = ""

        self.frequency = tk.StringVar(value="8")
        self.map_name = tk.StringVar(value="planet_sphere_f8")
        self.map_choice = tk.StringVar(value="")
        self.selected_tool = tk.StringVar(value="paint")
        self.rotation = tk.IntVar(value=0)
        self.status = tk.StringVar(value="正在准备球面编辑器")
        self.summary = tk.StringVar(value="尚未生成拓扑")
        self.selected_info = tk.StringVar(value="未选择格子")

        self._build_ui()
        self.refresh_brushes(show_dialog=False)
        self.window.after(50, self._poll_background)
        self.window.after(100, self.generate_layout)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=0)
        root.rowconfigure(0, weight=1)

        self._build_brush_panel(root)
        self._build_canvas_panel(root)
        self._build_control_panel(root)

        ttk.Label(
            self.window,
            textvariable=self.status,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=(8, 4),
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_brush_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="笔刷库", padding=8)
        frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="刷新笔刷库", command=self.refresh_brushes).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.brush_tree = ttk.Treeview(tree_frame, show="tree", height=24)
        self.brush_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.brush_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.brush_tree.configure(yscrollcommand=scrollbar.set)
        self.brush_tree.bind("<<TreeviewSelect>>", self._on_brush_selected)

        ttk.Label(frame, text="无效或缺失资源").grid(row=2, column=0, sticky="w", pady=(8, 3))
        self.issue_text = tk.Text(frame, width=32, height=9, wrap=tk.WORD, state=tk.DISABLED)
        self.issue_text.grid(row=3, column=0, sticky="ew")

    def _build_canvas_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="球面可视编辑视口", padding=6)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, background="#171b20", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<Button-1>", self._paint_cell)
        self.canvas.bind("<ButtonPress-3>", self._start_drag)
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-3>", self._end_drag)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom_steps(1.0))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom_steps(-1.0))

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="球面地图与工具", padding=10)
        frame.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="测试细分频率").grid(row=0, column=0, sticky="w")
        self.frequency_combo = ttk.Combobox(
            frame,
            textvariable=self.frequency,
            values=("2", "4", "8", "16", "32", "64", "128"),
            state="readonly",
            width=23,
        )
        self.frequency_combo.grid(row=1, column=0, sticky="ew", pady=(2, 5))
        self.frequency_combo.bind("<<ComboboxSelected>>", self._frequency_changed)
        self.generate_button = ttk.Button(frame, text="生成拓扑与分块", command=self.generate_layout)
        self.generate_button.grid(row=2, column=0, sticky="ew")

        ttk.Separator(frame).grid(row=3, column=0, sticky="ew", pady=9)
        ttk.Label(frame, text="新地图名称").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.map_name, width=25).grid(
            row=5, column=0, sticky="ew", pady=(2, 5)
        )
        self.create_button = ttk.Button(
            frame, text="新建球面地图", command=self.create_map, state=tk.DISABLED
        )
        self.create_button.grid(row=6, column=0, sticky="ew")

        ttk.Label(frame, text="打开兼容地图").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.map_combo = ttk.Combobox(frame, textvariable=self.map_choice, state="readonly", width=23)
        self.map_combo.grid(row=8, column=0, sticky="ew", pady=(2, 5))
        self.open_button = ttk.Button(frame, text="打开", command=self.open_map, state=tk.DISABLED)
        self.open_button.grid(row=9, column=0, sticky="ew")

        ttk.Separator(frame).grid(row=10, column=0, sticky="ew", pady=9)
        ttk.Radiobutton(
            frame, text="放置笔刷", variable=self.selected_tool, value="paint",
            command=self._sync_gpu_edit_state,
        ).grid(row=11, column=0, sticky="w")
        ttk.Radiobutton(
            frame, text="清除格子", variable=self.selected_tool, value="erase",
            command=self._sync_gpu_edit_state,
        ).grid(row=12, column=0, sticky="w")

        ttk.Label(frame, text="旋转方向").grid(row=13, column=0, sticky="w", pady=(8, 3))
        rotation_grid = ttk.Frame(frame)
        rotation_grid.grid(row=14, column=0, sticky="ew")
        for value in range(6):
            ttk.Radiobutton(
                rotation_grid,
                text=f"{value * 60}°",
                variable=self.rotation,
                value=value,
                command=self._update_selected_preview,
            ).grid(row=value // 2, column=value % 2, sticky="w", padx=(0, 8), pady=1)

        ttk.Label(frame, text="当前笔刷").grid(row=15, column=0, sticky="w", pady=(8, 3))
        self.preview_label = ttk.Label(frame, text="未选择", anchor=tk.CENTER)
        self.preview_label.grid(row=16, column=0, sticky="ew")
        self.selected_path_label = ttk.Label(frame, text="", wraplength=220, justify=tk.LEFT)
        self.selected_path_label.grid(row=17, column=0, sticky="ew", pady=(2, 0))

        self.save_button = ttk.Button(frame, text="保存脏区块", command=self.save_map, state=tk.DISABLED)
        self.save_button.grid(row=18, column=0, sticky="ew", pady=(9, 0))
        self.overview_button = ttk.Button(
            frame, text="生成/打开星球远景预览", command=self.open_planet_overview, state=tk.DISABLED
        )
        self.overview_button.grid(row=19, column=0, sticky="ew", pady=(5, 0))
        self.gpu_preview_button = ttk.Button(
            frame, text="打开 GPU 可编辑窗口", command=self.open_gpu_preview, state=tk.DISABLED
        )
        self.gpu_preview_button.grid(row=20, column=0, sticky="ew", pady=(5, 0))

        ttk.Separator(frame).grid(row=21, column=0, sticky="ew", pady=9)
        ttk.Label(frame, textvariable=self.summary, wraplength=235, justify=tk.LEFT).grid(
            row=22, column=0, sticky="ew"
        )
        ttk.Label(frame, textvariable=self.selected_info, wraplength=235, justify=tk.LEFT).grid(
            row=23, column=0, sticky="ew", pady=(7, 0)
        )

        ttk.Separator(frame).grid(row=24, column=0, sticky="ew", pady=9)
        ttk.Label(
            frame,
            text=(
                "左键：放置或清除\n"
                "右键拖动：旋转球体\n"
                "滚轮：缩放视口\n"
                "Ctrl+S：保存\n\n"
                "Pack 读取与解压在后台线程执行。\n"
                "笔刷纹理按 512/256/128/64 四级 LOD 后台生成。\n"
                "远距离使用分层区块可见性索引和区块边界投影，并可生成 512×256 星球缓存。\n"
                "Windows 可打开原生 OpenGL 3.3 GPU 编辑窗口；左键编辑、右键旋转、Ctrl+S 保存。纹理通过 2D Array 上传，六方向旋转在着色器执行。\n"
                "Tk 编辑视口仍保留为正确性与回退界面，测试频率开放到 frequency=128。"
            ),
            wraplength=235,
            justify=tk.LEFT,
        ).grid(row=25, column=0, sticky="w")

    def refresh_brushes(self, show_dialog: bool = True) -> None:
        try:
            self.scan_result = self.catalog.scan()
        except Exception as exc:
            messagebox.showerror("错误", f"刷新笔刷库失败：\n{exc}", parent=self.window)
            return
        if self.gpu_edit_bridge is not None:
            self._close_gpu_edit_bridge()
        self.records_by_uid = {record.uid: record for record in self.scan_result.records}
        self.preview_cache.clear()
        for cache in self.render_thumbnail_caches.values():
            cache.clear()
        self.distant_photo_cache.clear()
        self.distant_builder.set_context(self.session, self.topology, self.records_by_uid)
        self._populate_brush_tree()
        self._show_issues()
        self.planet_refresh_requested = self.session is not None and self.topology is not None
        self._request_planet_cache()
        self._redraw()
        message = (
            f"笔刷扫描完成：有效 {self.scan_result.active_count}，缺失 {self.scan_result.missing_count}，"
            f"无效 {len(self.scan_result.invalid)}"
        )
        self.status.set(message)
        if show_dialog:
            messagebox.showinfo("笔刷库", message, parent=self.window)

    def _populate_brush_tree(self) -> None:
        self.brush_tree.delete(*self.brush_tree.get_children())
        category_nodes: dict[str, str] = {"": ""}
        active_records = [] if self.scan_result is None else self.scan_result.active_records
        for record in sorted(active_records, key=lambda item: item.relative_path.casefold()):
            parent = ""
            cumulative: list[str] = []
            if record.category_path:
                for part in record.category_path.split("/"):
                    cumulative.append(part)
                    key = "/".join(cumulative)
                    if key not in category_nodes:
                        category_nodes[key] = self.brush_tree.insert(parent, tk.END, text=part, open=True)
                    parent = category_nodes[key]
            self.brush_tree.insert(
                parent,
                tk.END,
                text=Path(record.relative_path).name,
                tags=(f"uid:{record.uid}",),
            )

    def _show_issues(self) -> None:
        lines: list[str] = []
        if self.scan_result is not None:
            for invalid in self.scan_result.invalid:
                lines.append(f"无效：{invalid.relative_path}\n  {self._reason_text(invalid.reason)}")
            for record in self.scan_result.records:
                if record.state == "missing":
                    lines.append(f"缺失：{record.relative_path}\n  UID 保留，球面地图引用未清除")
        if not lines:
            lines.append("没有发现问题")
        self.issue_text.configure(state=tk.NORMAL)
        self.issue_text.delete("1.0", tk.END)
        self.issue_text.insert("1.0", "\n\n".join(lines))
        self.issue_text.configure(state=tk.DISABLED)

    @staticmethod
    def _reason_text(reason: str) -> str:
        if reason.startswith("invalid_size:"):
            return f"尺寸必须为 512×512，实际为 {reason.split(':', 1)[1]}"
        if reason.startswith("invalid_color_type:"):
            return "只支持 RGB 或 RGBA PNG"
        if reason.startswith("invalid_bit_depth:"):
            return "只支持 8 位 RGB/RGBA PNG"
        if reason == "brush_limit_exceeded":
            return "有效笔刷数量超过 4095 张"
        if reason == "not_png":
            return "文件扩展名为 PNG，但内容不是 PNG"
        return reason

    def _on_brush_selected(self, _event: tk.Event) -> None:
        selection = self.brush_tree.selection()
        if not selection:
            return
        tags = self.brush_tree.item(selection[0], "tags")
        uid = next((tag[4:] for tag in tags if tag.startswith("uid:")), None)
        if uid is None:
            return
        self.selected_brush_uid = uid
        self.selected_tool.set("paint")
        self._update_selected_preview()
        self._sync_gpu_edit_state()

    def _update_selected_preview(self) -> None:
        self._sync_gpu_edit_state()
        if self.selected_brush_uid is None:
            self.preview_label.configure(image="", text="未选择")
            self.selected_path_label.configure(text="")
            return
        record = self.records_by_uid.get(self.selected_brush_uid)
        if record is None or record.state != "active":
            image = self.preview_cache.get_missing(self.rotation.get())
            self.preview_label.configure(image=image, text="")
            self.preview_label.image = image
            self.selected_path_label.configure(text="笔刷文件缺失")
            return
        image = self.preview_cache.get(record.uid, record.relative_path, self.rotation.get())
        self.preview_label.configure(image=image, text="")
        self.preview_label.image = image
        self.selected_path_label.configure(text=record.relative_path)

    def _frequency_changed(self, _event: tk.Event) -> None:
        self.map_name.set(f"planet_sphere_f{self.frequency.get()}")
        self.status.set("频率已改变，点击“生成拓扑与分块”应用")

    def generate_layout(self) -> None:
        if self.generating:
            return
        if not self._confirm_discard_changes():
            return
        self._close_gpu_edit_bridge()
        try:
            frequency = int(self.frequency.get())
        except ValueError:
            messagebox.showerror("错误", "细分频率无效", parent=self.window)
            return
        self.generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.create_button.configure(state=tk.DISABLED)
        self.open_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.overview_button.configure(state=tk.DISABLED)
        self.gpu_preview_button.configure(state=tk.DISABLED)
        self.status.set(f"正在生成 frequency={frequency} 的拓扑与连通分块……")
        threading.Thread(target=self._generate_worker, args=(frequency,), daemon=True).start()
        self.window.after(50, self._poll_generation)

    def _generate_worker(self, frequency: int) -> None:
        try:
            topology = generate_dual_topology(frequency, max_cells=10 * 128 * 128 + 2)
            write_topology_cache(topology, self.paths.map_root)
            layout = build_chunk_layout(topology, target_cells=256)
            write_chunk_layout_cache(layout, self.paths.map_root / ".topology")
            visibility_index = build_chunk_visibility_index(topology, layout)
            write_chunk_visibility_cache(visibility_index, self.paths.map_root / ".topology")
            self.generation_results.put(("success", (topology, layout, visibility_index)))
        except Exception as exc:
            self.generation_results.put(("error", exc))

    def _poll_generation(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value = self.generation_results.get_nowait()
        except queue.Empty:
            if self.generating:
                self.window.after(50, self._poll_generation)
            return
        self.generating = False
        self.generate_button.configure(state=tk.NORMAL)
        if state == "error":
            self.status.set("拓扑与分块生成失败")
            messagebox.showerror("生成失败", str(value), parent=self.window)
            return
        topology, layout, visibility_index = value
        self.topology = topology
        self.layout = layout
        self.visibility_index = visibility_index
        self.session = None
        self.chunk_loader.set_session(None)
        self.distant_builder.set_context(None, topology, self.records_by_uid)
        self.lod_policy = BrushLodPolicy(initial_level=0)
        self.current_lod_level = 0
        self.selected_cell_id = None
        self.planet_refresh_requested = False
        self.viewport = SphereViewport()
        self.create_button.configure(state=tk.NORMAL)
        self._refresh_map_choices()
        self.status.set(
            f"拓扑、分块与层级可见性已准备：Cell {layout.cell_count:,}，区块 {layout.chunk_count:,}，索引节点 {visibility_index.node_count:,}"
        )
        self._redraw()

    def _refresh_map_choices(self) -> None:
        layout = self.layout
        names = [] if layout is None else self.store.list_compatible_maps(layout)
        self.map_combo.configure(values=names)
        if names:
            if self.map_choice.get() not in names:
                self.map_choice.set(names[0])
            self.open_button.configure(state=tk.NORMAL)
        else:
            self.map_choice.set("")
            self.open_button.configure(state=tk.DISABLED)

    def create_map(self) -> None:
        layout = self.layout
        if layout is None:
            return
        if not self._confirm_discard_changes():
            return
        self._close_gpu_edit_bridge()
        try:
            session = self.store.create_blank(self.map_name.get().strip(), layout)
        except (SphereMapError, OSError) as exc:
            messagebox.showerror("新建失败", str(exc), parent=self.window)
            return
        self.session = session
        self.chunk_loader.set_session(session)
        self.distant_builder.set_context(session, self.topology, self.records_by_uid)
        self.selected_cell_id = None
        self.save_button.configure(state=tk.NORMAL)
        self.overview_button.configure(state=tk.NORMAL)
        self.gpu_preview_button.configure(state=tk.NORMAL)
        self._refresh_map_choices()
        self.status.set(f"已新建球面地图：{session.name}")
        self.planet_refresh_requested = True
        self._request_planet_cache()
        self._redraw()

    def open_map(self) -> None:
        layout = self.layout
        name = self.map_choice.get().strip()
        if layout is None or not name:
            return
        if not self._confirm_discard_changes():
            return
        self._close_gpu_edit_bridge()
        try:
            session = self.store.open(name, layout)
        except (SphereMapError, OSError) as exc:
            messagebox.showerror("打开失败", str(exc), parent=self.window)
            return
        self.session = session
        self.chunk_loader.set_session(session)
        self.distant_builder.set_context(session, self.topology, self.records_by_uid)
        self.map_name.set(session.name)
        self.selected_cell_id = None
        self.save_button.configure(state=tk.NORMAL)
        self.overview_button.configure(state=tk.NORMAL)
        self.gpu_preview_button.configure(state=tk.NORMAL)
        self.status.set(f"已打开球面地图：{session.name}")
        self.planet_refresh_requested = True
        self._request_planet_cache()
        self._redraw()

    def _current_gpu_tool_state(self) -> GpuToolState:
        uid = self.selected_brush_uid
        record = None if uid is None else self.records_by_uid.get(uid)
        path = "" if record is None else record.relative_path
        return GpuToolState(
            tool=self.selected_tool.get(),
            brush_uid=uid,
            last_known_path=path,
            rotation=self.rotation.get(),
        )

    def _sync_gpu_edit_state(self) -> None:
        bridge = self.gpu_edit_bridge
        if bridge is None or bridge.closed:
            return
        try:
            bridge.set_tool_state(self._current_gpu_tool_state())
        except ValueError as exc:
            self.status.set(f"GPU 工具状态无效：{exc}")

    def _close_gpu_edit_bridge(self) -> None:
        bridge = self.gpu_edit_bridge
        if bridge is not None:
            bridge.close()
        self.gpu_edit_bridge = None
        self.gpu_edit_lod_level = None

    def _poll_gpu_edit_requests(self) -> bool:
        bridge = self.gpu_edit_bridge
        if bridge is None:
            return False
        if bridge.closed:
            self.gpu_edit_bridge = None
            self.gpu_edit_lod_level = None
            return False

        bridge_state = bridge.tool_state()
        preview_changed = False
        if self.selected_tool.get() != bridge_state.tool:
            self.selected_tool.set(bridge_state.tool)
        if self.rotation.get() != bridge_state.rotation:
            self.rotation.set(bridge_state.rotation)
            preview_changed = True
        if preview_changed:
            self._update_selected_preview()

        changed = False
        for request in bridge.poll_requests():
            if isinstance(request, GpuSaveRequest):
                self._handle_gpu_save_request(bridge, request)
                continue
            if isinstance(request, GpuEditRequest):
                if self._handle_gpu_edit_request(bridge, request):
                    changed = True
        return changed

    def _handle_gpu_edit_request(
        self, bridge: GpuEditBridge, request: GpuEditRequest
    ) -> bool:
        if request.phase == "end":
            bridge.push_patch(
                GpuStatusPatch(request.request_id, True, "连续笔划结束")
            )
            return False
        session = self.session
        topology = self.topology
        lod_level = self.gpu_edit_lod_level
        if session is None or topology is None or lod_level is None:
            bridge.push_patch(GpuStatusPatch(request.request_id, False, "当前地图会话已经关闭"))
            return False
        if request.cell_id in set(topology.pentagon_ids):
            bridge.push_patch(
                GpuStatusPatch(request.request_id, False, f"CellId {request.cell_id} 是隐藏五边形")
            )
            return False

        state = request.state
        try:
            chunk_id, _local_index = session.layout.chunk_for_cell(request.cell_id)
            self.chunk_loader.ensure_loaded(session, chunk_id)
            if state.tool == "erase":
                session.set_cell(request.cell_id, self.store, None)
                texture_key = EMPTY_LAYER_KEY
                rotation = 0
                pixels = None
                width = 0
                height = 0
                action = "清除"
            else:
                if state.brush_uid is None:
                    raise SphereMapError("请先在主窗口选择一张有效笔刷")
                record = self.records_by_uid.get(state.brush_uid)
                if record is None or record.state != "active":
                    raise SphereMapError("当前笔刷文件缺失，不能放置")
                session.set_cell(
                    request.cell_id,
                    self.store,
                    record.uid,
                    record.relative_path,
                    state.rotation,
                )
                texture_key = brush_texture_key(record, lod_level)
                rotation = state.rotation
                pixels = None
                width = 0
                height = 0
                if bridge.claim_texture_key(texture_key):
                    try:
                        payload = build_brush_texture_payload(
                            self.paths.brush_root, record, lod_level
                        )
                    except Exception:
                        bridge.forget_texture_key(texture_key)
                        raise
                    pixels = payload.pixels_rgba
                    width = payload.width
                    height = payload.height
                action = "放置"
        except (SphereMapError, OSError, ValueError) as exc:
            bridge.push_patch(GpuStatusPatch(request.request_id, False, f"编辑失败：{exc}"))
            self.status.set(f"GPU 编辑失败：{exc}")
            return False

        self.selected_cell_id = request.cell_id
        self.distant_builder.invalidate((chunk_id,))
        message = f"已{action} CellId {request.cell_id}，Chunk {chunk_id} 已标记为脏区块"
        bridge.push_patch(
            GpuCellPatch(
                request_id=request.request_id,
                cell_id=request.cell_id,
                texture_key=texture_key,
                rotation=rotation,
                message=message,
                pixels_rgba=pixels,
                texture_width=width,
                texture_height=height,
            )
        )
        self.status.set(message)
        return True

    def _push_gpu_cell_refresh(self, cell_id: int, message: str) -> None:
        bridge = self.gpu_edit_bridge
        session = self.session
        lod_level = self.gpu_edit_lod_level
        if bridge is None or bridge.closed or session is None or lod_level is None:
            return
        try:
            uid, rotation = session.brush_uid_for_cell(cell_id, self.store)
            pixels = None
            width = 0
            height = 0
            if uid is None:
                texture_key = EMPTY_LAYER_KEY
                rotation = 0
            else:
                record = self.records_by_uid.get(uid)
                if record is None or record.state != "active":
                    texture_key = MISSING_LAYER_KEY
                else:
                    texture_key = brush_texture_key(record, lod_level)
                    if bridge.claim_texture_key(texture_key):
                        try:
                            payload = build_brush_texture_payload(
                                self.paths.brush_root, record, lod_level
                            )
                        except Exception:
                            bridge.forget_texture_key(texture_key)
                            raise
                        pixels = payload.pixels_rgba
                        width = payload.width
                        height = payload.height
            bridge.push_patch(
                GpuCellPatch(
                    request_id=0,
                    cell_id=cell_id,
                    texture_key=texture_key,
                    rotation=rotation,
                    message=message,
                    pixels_rgba=pixels,
                    texture_width=width,
                    texture_height=height,
                )
            )
        except (SphereMapError, OSError, ValueError) as exc:
            bridge.push_patch(GpuStatusPatch(0, False, f"同步失败：{exc}"))

    def _handle_gpu_save_request(
        self, bridge: GpuEditBridge, request: GpuSaveRequest
    ) -> None:
        session = self.session
        if session is None:
            bridge.push_patch(GpuStatusPatch(request.request_id, False, "当前没有已打开的球面地图"))
            return
        dirty_chunk_ids = tuple(sorted(session.dirty_chunks))
        dirty_before = len(dirty_chunk_ids)
        try:
            self.store.save(session)
            self.distant_builder.invalidate(dirty_chunk_ids)
            if self.projection is not None:
                required = (
                    set()
                    if self.current_lod_level == 4
                    else set(self.projection.visible_chunk_ids)
                )
                self.chunk_loader.request(session, required)
            if self.topology is not None:
                self.planet_refresh_requested = True
                self._request_planet_cache()
        except (SphereMapError, OSError) as exc:
            message = f"保存失败：{exc}"
            bridge.push_patch(GpuStatusPatch(request.request_id, False, message))
            self.status.set(message)
            return
        message = f"保存完成：写入 {dirty_before} 个脏区块"
        bridge.push_patch(GpuStatusPatch(request.request_id, True, message))
        self.status.set(message)

    def open_gpu_preview(self) -> None:
        if self.gpu_preview_pending:
            self.status.set("GPU 批量数据正在生成")
            return
        topology = self.topology
        layout = self.layout
        session = self.session
        projection = self.projection
        if topology is None or layout is None or session is None or projection is None:
            self.status.set("请先生成拓扑并新建或打开球面地图")
            return
        if self.current_lod_level > 3 or not projection.visible_cell_ids:
            messagebox.showinfo(
                "GPU 编辑器",
                "当前处于区块远景模式，请放大到逐格 LOD 后再打开 GPU 编辑器。",
                parent=self.window,
            )
            return
        if not native_gpu_supported():
            messagebox.showinfo(
                "GPU 编辑器",
                "原生 OpenGL 预览当前只在 Windows 上启用。数据批处理与纹理数组测试仍可在其他平台运行。",
                parent=self.window,
            )
            return

        complete_test_sphere = topology.cell_count <= 200_000
        cell_ids = (
            tuple(range(topology.cell_count))
            if complete_test_sphere
            else tuple(projection.visible_cell_ids)
        )
        records = dict(self.records_by_uid)
        lod_level = self.current_lod_level
        yaw = self.viewport.yaw
        pitch = self.viewport.pitch
        zoom = self.viewport.zoom
        self.gpu_preview_pending = True
        self.gpu_preview_button.configure(state=tk.DISABLED)
        scope_text = "完整测试星球" if complete_test_sphere else "当前视口快照"
        self.status.set(
            f"正在生成 GPU 实例批次：{scope_text}，候选 {len(cell_ids):,} 格，LOD{lod_level}……"
        )

        def worker() -> None:
            try:
                batch = self.gpu_batch_builder.build(
                    topology,
                    layout,
                    session,
                    self.store,
                    records,
                    cell_ids,
                    lod_level,
                )
                self.gpu_preview_results.put(("success", (batch, topology, yaw, pitch, zoom)))
            except Exception as exc:
                self.gpu_preview_results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-gpu-batch", daemon=True).start()

    def _poll_gpu_preview(self) -> None:
        try:
            state, value = self.gpu_preview_results.get_nowait()
        except queue.Empty:
            return
        self.gpu_preview_pending = False
        if self.session is not None:
            self.gpu_preview_button.configure(state=tk.NORMAL)
        if state == "error":
            self.status.set("GPU 实例批次生成失败")
            messagebox.showerror("GPU 编辑器失败", str(value), parent=self.window)
            return
        batch, topology, yaw, pitch, zoom = value
        self._close_gpu_edit_bridge()
        initial_state = self._current_gpu_tool_state()
        bridge = GpuEditBridge(
            initial_state=initial_state,
            known_texture_keys=(layer.key for layer in batch.texture_layers),
        )
        self.gpu_edit_bridge = bridge
        self.gpu_edit_lod_level = batch.lod_level

        def report_error(message: str) -> None:
            try:
                self.window.after(0, lambda: messagebox.showerror("GPU 编辑器失败", message, parent=self.window))
            except tk.TclError:
                return

        launch = launch_gpu_preview(
            batch,
            yaw,
            pitch,
            zoom,
            topology=topology,
            edit_bridge=bridge,
            title=f"Hex Planet GPU Editor - {batch.instance_count:,} instances / {batch.texture_layer_count} layers",
            on_error=report_error,
        )
        if not launch.started:
            bridge.close()
            self.gpu_edit_bridge = None
            messagebox.showerror("GPU 编辑器失败", launch.reason, parent=self.window)
            return
        self.status.set(
            f"GPU 可编辑窗口已启动：实例 {batch.instance_count:,}，纹理数组层 {batch.texture_layer_count}，LOD{batch.lod_level}"
        )

    def save_map(self) -> None:
        session = self.session
        if session is None:
            self.status.set("当前没有已打开的球面地图")
            return
        dirty_chunk_ids = tuple(sorted(session.dirty_chunks))
        dirty_before = len(dirty_chunk_ids)
        try:
            self.store.save(session)
            self.distant_builder.invalidate(dirty_chunk_ids)
            if self.projection is not None:
                required = (
                    set()
                    if self.current_lod_level == 4
                    else set(self.projection.visible_chunk_ids)
                )
                self.chunk_loader.request(session, required)
            if self.topology is not None:
                self.planet_refresh_requested = True
                self._request_planet_cache()
        except (SphereMapError, OSError) as exc:
            if self.gpu_edit_bridge is not None:
                self.gpu_edit_bridge.push_patch(GpuStatusPatch(0, False, f"保存失败：{exc}"))
            messagebox.showerror("保存失败", str(exc), parent=self.window)
            return
        message = f"保存完成：写入 {dirty_before} 个脏区块"
        if self.gpu_edit_bridge is not None:
            self.gpu_edit_bridge.push_patch(GpuStatusPatch(0, True, message))
        self.status.set(message)
        self._redraw()

    def _request_planet_cache(self) -> None:
        session = self.session
        topology = self.topology
        if not self.planet_refresh_requested or session is None or topology is None:
            return
        if session.dirty_chunks or session.brush_table_dirty:
            return
        info = self.planet_builder.request(session, topology, self.records_by_uid)
        if info is not None:
            self.planet_refresh_requested = False
            if self.overview_requested:
                self._show_planet_overview(info.path)

    def open_planet_overview(self) -> None:
        session = self.session
        topology = self.topology
        if session is None or topology is None:
            self.status.set("请先新建或打开球面地图")
            return
        if session.dirty_chunks or session.brush_table_dirty:
            self.status.set("星球远景缓存只基于已保存 Pack，请先保存当前修改")
            return
        self.overview_requested = True
        self.planet_refresh_requested = True
        self._ensure_overview_window("正在后台生成星球远景缓存……")
        self._request_planet_cache()

    def _ensure_overview_window(self, text: str = "") -> None:
        if self.overview_window is not None and self.overview_window.winfo_exists():
            if self.overview_label is not None and text:
                self.overview_label.configure(image="", text=text)
            self.overview_window.lift()
            return
        window = tk.Toplevel(self.window)
        window.title("星球远景缓存预览 512×256")
        window.resizable(False, False)
        label = ttk.Label(window, text=text, padding=12, anchor=tk.CENTER)
        label.pack(fill=tk.BOTH, expand=True)
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        self.overview_window = window
        self.overview_label = label

    def _show_planet_overview(self, path: Path) -> None:
        self._ensure_overview_window()
        if self.overview_label is None:
            return
        try:
            image = tk.PhotoImage(master=self.overview_window, file=str(path))
        except tk.TclError as exc:
            self.overview_label.configure(image="", text=f"无法读取远景缓存：{exc}")
            return
        self.overview_image = image
        self.overview_label.configure(image=image, text="")
        self.overview_label.image = image
        self.overview_requested = False

    def _paint_cell(self, event: tk.Event) -> None:
        topology = self.topology
        session = self.session
        projection = self.projection
        if topology is None or session is None or projection is None:
            self.status.set("请先生成拓扑并新建或打开球面地图")
            return
        if self.current_lod_level == 4:
            self.status.set("当前是区块远景模式，未逐格投影；请放大后再绘制")
            return
        cell_id = projection.hit_test(event.x, event.y)
        if cell_id is None:
            return
        if cell_id in set(topology.pentagon_ids):
            self.status.set(f"CellId {cell_id} 是隐藏五边形，不能绘制")
            return
        try:
            chunk_id, _local_index = session.layout.chunk_for_cell(cell_id)
            self.chunk_loader.ensure_loaded(session, chunk_id)
            if self.selected_tool.get() == "erase":
                session.set_cell(cell_id, self.store, None)
            else:
                if self.selected_brush_uid is None:
                    self.status.set("请先选择一张有效笔刷")
                    return
                record = self.records_by_uid.get(self.selected_brush_uid)
                if record is None or record.state != "active":
                    self.status.set("当前笔刷文件缺失，不能放置")
                    return
                session.set_cell(
                    cell_id,
                    self.store,
                    record.uid,
                    record.relative_path,
                    self.rotation.get(),
                )
        except SphereMapError as exc:
            messagebox.showerror("编辑失败", str(exc), parent=self.window)
            return
        self.selected_cell_id = cell_id
        self.distant_builder.invalidate((chunk_id,))
        message = f"已修改 CellId {cell_id}，所在区块已标记为脏区块"
        self._push_gpu_cell_refresh(cell_id, message)
        self.status.set(message)
        self._redraw()

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_origin = (event.x, event.y, self.viewport.yaw, self.viewport.pitch)

    def _drag(self, event: tk.Event) -> None:
        if self.drag_origin is None:
            return
        start_x, start_y, start_yaw, start_pitch = self.drag_origin
        self.viewport.yaw = start_yaw + (event.x - start_x) * 0.008
        self.viewport.pitch = max(
            -1.45,
            min(1.45, start_pitch + (event.y - start_y) * 0.008),
        )
        self._redraw()

    def _end_drag(self, _event: tk.Event) -> None:
        self.drag_origin = None

    def _mouse_wheel(self, event: tk.Event) -> None:
        steps = 1.0 if event.delta > 0 else -1.0
        self._zoom_steps(steps)

    def _zoom_steps(self, steps: float) -> None:
        self.viewport.zoom_by(steps)
        self._redraw()

    def _redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        self.render_images.clear()
        topology = self.topology
        layout = self.layout
        visibility = self.visibility_index
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if topology is None or layout is None or visibility is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="正在准备球面拓扑、分块与层级可见性索引……",
                fill="#c8d0d8",
                font=("TkDefaultFont", 13),
            )
            return

        visibility_result = visibility.query(
            layout,
            self.viewport.yaw,
            self.viewport.pitch,
            self.viewport.zoom,
            width,
            height,
        )
        self.current_lod_level = self.lod_policy.update(visibility_result.candidate_cells)
        distant_mode = self.current_lod_level == 4
        if distant_mode:
            projection = self.viewport.project_chunks(
                topology,
                visibility,
                visibility_result.chunk_ids,
                width,
                height,
                candidate_cell_count=visibility_result.candidate_cells,
                visited_visibility_nodes=visibility_result.visited_nodes,
                tested_visibility_chunks=visibility_result.tested_chunks,
            )
        else:
            candidate_cells = visibility.candidate_cell_ids(layout, visibility_result.chunk_ids)
            projection = self.viewport.project(
                topology,
                width,
                height,
                layout,
                cell_ids=candidate_cells,
                candidate_cell_count=visibility_result.candidate_cells,
                visited_visibility_nodes=visibility_result.visited_nodes,
                tested_visibility_chunks=visibility_result.tested_chunks,
            )
            self.current_lod_level = self.lod_policy.update(len(projection.visible_cell_ids))
            distant_mode = self.current_lod_level == 4
            if distant_mode:
                projection = self.viewport.project_chunks(
                    topology,
                    visibility,
                    visibility_result.chunk_ids,
                    width,
                    height,
                    candidate_cell_count=visibility_result.candidate_cells,
                    visited_visibility_nodes=visibility_result.visited_nodes,
                    tested_visibility_chunks=visibility_result.tested_chunks,
                )
        self.projection = projection
        self.canvas.create_oval(
            projection.sphere_center_x - projection.sphere_radius,
            projection.sphere_center_y - projection.sphere_radius,
            projection.sphere_center_x + projection.sphere_radius,
            projection.sphere_center_y + projection.sphere_radius,
            fill="#222a31",
            outline="#687480",
            width=2,
        )

        session = self.session
        unloaded: tuple[int, ...] = ()
        retained_dirty: tuple[int, ...] = ()
        if session is not None:
            try:
                detailed_chunks = set() if distant_mode else set(projection.visible_chunk_ids)
                request_update = self.chunk_loader.request(session, detailed_chunks)
                unloaded = request_update.unloaded
                retained_dirty = request_update.retained_dirty
                if distant_mode:
                    self.distant_builder.request(
                        session, topology, self.records_by_uid, projection.visible_chunk_ids
                    )
                else:
                    self.distant_builder.request(session, topology, self.records_by_uid, ())
                self.last_viewport_error = ""
            except (SphereMapError, RuntimeError, ValueError) as exc:
                message = str(exc)
                if message != self.last_viewport_error:
                    self.status.set(f"视口区块请求失败：{message}")
                    self.last_viewport_error = message

        if distant_mode:
            self._draw_distant_chunks(projection, layout)
        else:
            self._draw_detail_cells(projection, topology)

        loaded_count = 0 if session is None else len(session.loaded_chunks)
        dirty_count = 0 if session is None else len(session.dirty_chunks)
        visible_cell_text = (
            f"详细可见格子：{len(projection.visible_cell_ids):,}"
            if not distant_mode
            else f"远景候选格子（未逐格投影）：{projection.candidate_cell_count:,}"
        )
        self.summary.set(
            "\n".join(
                (
                    f"Cell：{layout.cell_count:,}  区块：{layout.chunk_count:,}",
                    visible_cell_text,
                    f"当前可见区块：{len(projection.visible_chunk_ids):,}",
                    f"层级查询：访问节点 {projection.visited_visibility_nodes:,}  精测区块 {projection.tested_visibility_chunks:,}",
                    f"活动区块：{loaded_count}  脏区块：{dirty_count}",
                    f"后台区块读取：{len(self.chunk_loader.pending)}",
                    f"纹理：{BrushLodPolicy.description(self.current_lod_level)}",
                    f"后台 LOD 生成：{self.lod_builder.pending_count()}",
                    f"远景区块缓存：就绪 {len(self.distant_builder.ready)}  后台 {self.distant_builder.pending_count()}",
                    f"星球远景缓存：{'生成中' if self.planet_builder.is_pending() else '就绪/待命'}",
                    f"缩放：{self.viewport.zoom:.2f}×",
                )
            )
        )
        self._update_selected_info()
        if session is not None and (unloaded or retained_dirty):
            self.status.set(
                f"视口更新：后台读取 {len(self.chunk_loader.pending)}，"
                f"释放 {len(unloaded)}，保留脏区块 {len(retained_dirty)}"
            )

    def _draw_detail_cells(
        self, projection: ViewportProjection, topology: DualTopology
    ) -> None:
        pentagons = set(topology.pentagon_ids)
        for cell in projection.cells:
            if cell.cell_id in pentagons:
                continue
            uid, rotation = self._cell_brush_state(cell.cell_id)
            fill = "#323a42" if uid is None else "#596671"
            outline = self._shade("#8b98a4", cell.depth)
            self.canvas.create_polygon(cell.points, fill=fill, outline="", width=0)
            if uid is not None and cell.depth >= 0.48:
                image = self._render_image_for_cell(
                    cell, uid, rotation, self.current_lod_level
                )
                if image is not None:
                    self.render_images.append(image)
                    self.canvas.create_image(cell.center_x, cell.center_y, image=image)
            width_value = 2 if cell.cell_id == self.selected_cell_id else 1
            outline_value = "#f0c36b" if cell.cell_id == self.selected_cell_id else outline
            self.canvas.create_polygon(
                cell.points,
                fill="",
                outline=outline_value,
                width=width_value,
            )

    def _draw_distant_chunks(
        self,
        projection: ViewportProjection,
        layout: ChunkLayout,
    ) -> None:
        selected_chunk = None
        if self.selected_cell_id is not None:
            selected_chunk = layout.cell_to_chunk[self.selected_cell_id]

        for chunk in projection.chunks:
            points = chunk.points
            info = self.distant_builder.info(chunk.chunk_id)
            base = "#3b4650" if info is None else self._rgb_hex(info.average_rgb)
            fill = self._shade(base, chunk.depth)
            outline = (
                "#f0c36b"
                if chunk.chunk_id == selected_chunk
                else self._shade("#71808c", chunk.depth)
            )
            width_value = 2 if chunk.chunk_id == selected_chunk else 1
            self.canvas.create_polygon(points, fill=fill, outline=outline, width=width_value)
            if info is None or chunk.depth < 0.45:
                continue
            target = min(chunk.max_x - chunk.min_x, chunk.max_y - chunk.min_y) * 0.72
            image = self._distant_photo(info.signature, info.path, target)
            if image is not None:
                center_x, center_y = polygon_center(points)
                self.render_images.append(image)
                self.canvas.create_image(center_x, center_y, image=image)

    def _distant_photo(
        self, signature: str, path: Path, target_size: float
    ) -> tk.PhotoImage | None:
        if target_size >= 28:
            factor = 2
        elif target_size >= 14:
            factor = 4
        elif target_size >= 7:
            factor = 8
        else:
            return None
        key = (signature, factor)
        cached = self.distant_photo_cache.get(key)
        if cached is not None:
            return cached
        try:
            source = tk.PhotoImage(master=self.window, file=str(path))
            image = source.subsample(factor, factor)
        except tk.TclError:
            return None
        self.distant_photo_cache[key] = image
        return image

    @staticmethod
    def _rgb_hex(color: tuple[int, int, int]) -> str:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    def _poll_background(self) -> None:
        if not self.window.winfo_exists():
            return
        self._poll_gpu_preview()
        should_redraw = self._poll_gpu_edit_requests()
        session = self.session
        if session is not None:
            update = self.chunk_loader.poll(session)
            if update.loaded or update.discarded:
                should_redraw = True
            if update.errors:
                chunk_id, message = update.errors[0]
                self.status.set(f"后台区块读取失败：Chunk {chunk_id}：{message}")
        lod_results = self.lod_builder.poll()
        if lod_results:
            should_redraw = True
            for result in lod_results:
                if result.error:
                    self.status.set(f"笔刷 LOD 生成失败：{result.error}")
                    break
        distant_results = self.distant_builder.poll()
        if distant_results:
            should_redraw = True
            for result in distant_results:
                if result.error:
                    self.status.set(f"区块远景缓存生成失败：Chunk {result.chunk_id}：{result.error}")
                    break
        planet_result = self.planet_builder.poll()
        if planet_result is not None:
            if planet_result.error:
                self.status.set(f"星球远景缓存生成失败：{planet_result.error}")
            else:
                self.status.set("星球远景缓存已生成")
            self._request_planet_cache()
            should_redraw = True
        elif self.planet_refresh_requested and not self.planet_builder.is_pending():
            self._request_planet_cache()
        if should_redraw:
            self._redraw()
        self.window.after(50, self._poll_background)

    def _cell_brush_state(self, cell_id: int) -> tuple[str | None, int]:
        session = self.session
        layout = self.layout
        if session is None or layout is None:
            return None, 0
        chunk_id, local_index = layout.chunk_for_cell(cell_id)
        values = session.loaded_chunks.get(chunk_id)
        if values is None:
            return None, 0
        value = values[local_index]
        local_id = value & 0x0FFF
        rotation = (value >> 12) & 0x0007
        if local_id == 0:
            return None, rotation
        entry = session.brush_entries.get(local_id)
        return (None, rotation) if entry is None else (entry.brush_uid, rotation)

    def _render_image_for_cell(
        self, cell: ProjectedCell, uid: str, rotation: int, lod_level: int
    ) -> tk.PhotoImage | None:
        width = cell.max_x - cell.min_x
        height = cell.max_y - cell.min_y
        target_side = min(width / 2.0, height / max(1.0, math.sqrt(3))) * 0.86
        side = self._quantized_side(target_side)
        if side is None:
            return None
        cache = self.render_thumbnail_caches.get(side)
        if cache is None:
            cache = BrushThumbnailCache(self.window, self.paths.brush_root, side=side)
            self.render_thumbnail_caches[side] = cache
        record = self.records_by_uid.get(uid)
        if record is None or record.state != "active":
            return cache.get_missing(rotation)
        lod_path = self.lod_builder.request(record, lod_level)
        if lod_path is None:
            return None
        cache_key = f"{record.uid}:{record.content_hash}:lod{lod_level}"
        return cache.get_path(cache_key, lod_path, rotation, padding=LOD_PADDING)

    @staticmethod
    def _quantized_side(target: float) -> int | None:
        sizes = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48)
        valid = [size for size in sizes if size <= target]
        return max(valid) if valid else None

    def _update_selected_info(self) -> None:
        cell_id = self.selected_cell_id
        topology = self.topology
        layout = self.layout
        if cell_id is None or topology is None or layout is None:
            self.selected_info.set("未选择格子")
            return
        chunk_id, local_index = layout.chunk_for_cell(cell_id)
        uid, rotation = self._cell_brush_state(cell_id)
        self.selected_info.set(
            "\n".join(
                (
                    f"CellId：{cell_id}",
                    f"ChunkId：{chunk_id}  局部序号：{local_index}",
                    f"笔刷 UID：{uid or '空'}",
                    f"旋转：{rotation * 60}°",
                )
            )
        )

    @staticmethod
    def _shade(base: str, depth: float) -> str:
        red = int(base[1:3], 16)
        green = int(base[3:5], 16)
        blue = int(base[5:7], 16)
        factor = 0.42 + 0.58 * max(0.0, min(1.0, depth))
        return f"#{int(red * factor):02x}{int(green * factor):02x}{int(blue * factor):02x}"

    def _confirm_discard_changes(self) -> bool:
        session = self.session
        if session is None or (not session.dirty_chunks and not session.brush_table_dirty):
            return True
        return messagebox.askyesno(
            "尚未保存",
            "当前球面地图有未保存修改，确定继续吗？",
            parent=self.window,
        )

    def _close_workers(self) -> None:
        self._close_gpu_edit_bridge()
        self.chunk_loader.close()
        self.lod_builder.close()
        self.distant_builder.close()
        self.planet_builder.close()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self.window:
            self._close_workers()

    def _on_close(self) -> None:
        if self._confirm_discard_changes():
            self._close_workers()
            self.window.destroy()
