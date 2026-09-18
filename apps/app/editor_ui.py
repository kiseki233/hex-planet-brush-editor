from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .acceptance_viewer import AcceptanceViewer
from .brush_catalog import BrushCatalog, BrushRecord, ScanResult
from .chunk_storage_viewer import ChunkStorageViewer
from .map_store import LocalMap, MapFormatError, MapStore
from .sphere_editor import SphereMapEditor
from .paths import ProjectPaths
from .production_builder import ProductionBuilder
from .production_editor import ProductionSphereEditor
from .thumbnail import BrushThumbnailCache
from .topology_viewer import TopologyViewer


class EditorApp:
    def __init__(self, root: tk.Tk, paths: ProjectPaths) -> None:
        self.root = root
        self.paths = paths
        self.catalog = BrushCatalog(paths.brush_root)
        self.map_store = MapStore(paths.map_root)
        self.scan_result: ScanResult | None = None
        self.records_by_uid: dict[str, BrushRecord] = {}
        self.current_map = self.map_store.create_blank("planet_local_001")
        self.selected_brush_uid: str | None = None
        self.selected_tool = tk.StringVar(value="paint")
        self.rotation = tk.IntVar(value=0)
        self.map_name = tk.StringVar(value=self.current_map.name)
        self.map_choice = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="准备就绪")
        self.side = 24
        self.hex_width = self.side * 2
        self.hex_height = max(2, round(math.sqrt(3) * self.side))
        self.margin = 18
        self.cell_centers: list[tuple[float, float]] = []
        self.thumbnail_cache = BrushThumbnailCache(root, paths.brush_root, side=self.side)

        self._configure_root()
        self._build_ui()
        self.refresh_brushes(show_dialog=False)
        self._refresh_map_choices()
        self._redraw_map()

    def _configure_root(self) -> None:
        self.root.title("六边形星球地图笔刷编辑器 v1.3.1")
        self.root.geometry("1280x820")
        self.root.minsize(1020, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-s>", lambda _event: self.save_map())

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0)
        container.rowconfigure(0, weight=1)

        self._build_brush_panel(container)
        self._build_canvas_panel(container)
        self._build_control_panel(container)

        status = ttk.Label(self.root, textvariable=self.status_text, anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_brush_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="笔刷库", padding=8)
        frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="刷新笔刷库", command=self.refresh_brushes).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.brush_tree = ttk.Treeview(tree_frame, show="tree", height=22)
        self.brush_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.brush_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.brush_tree.configure(yscrollcommand=tree_scroll.set)
        self.brush_tree.bind("<<TreeviewSelect>>", self._on_brush_selected)

        ttk.Label(frame, text="无效或缺失资源").grid(row=2, column=0, sticky="w", pady=(8, 3))
        self.issue_text = tk.Text(frame, width=34, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.issue_text.grid(row=3, column=0, sticky="ew")

    def _build_canvas_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="16×16 局部六边形编辑原型", padding=6)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, background="#20242a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        horizontal = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="地图与工具", padding=10)
        frame.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="地图名称").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.map_name, width=25).grid(row=1, column=0, sticky="ew", pady=(2, 8))

        button_row = ttk.Frame(frame)
        button_row.grid(row=2, column=0, sticky="ew")
        ttk.Button(button_row, text="新建", command=self.new_map).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(button_row, text="保存", command=self.save_map).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        ttk.Separator(frame).grid(row=3, column=0, sticky="ew", pady=10)
        ttk.Label(frame, text="打开已有地图").grid(row=4, column=0, sticky="w")
        self.map_combo = ttk.Combobox(frame, textvariable=self.map_choice, state="readonly", width=23)
        self.map_combo.grid(row=5, column=0, sticky="ew", pady=(2, 5))
        ttk.Button(frame, text="打开", command=self.open_map).grid(row=6, column=0, sticky="ew")

        ttk.Separator(frame).grid(row=7, column=0, sticky="ew", pady=10)
        ttk.Label(frame, text="编辑工具").grid(row=8, column=0, sticky="w")
        ttk.Radiobutton(frame, text="放置笔刷", variable=self.selected_tool, value="paint").grid(row=9, column=0, sticky="w")
        ttk.Radiobutton(frame, text="清除格子", variable=self.selected_tool, value="erase").grid(row=10, column=0, sticky="w")

        ttk.Label(frame, text="旋转方向").grid(row=11, column=0, sticky="w", pady=(10, 3))
        rotation_grid = ttk.Frame(frame)
        rotation_grid.grid(row=12, column=0, sticky="ew")
        for value in range(6):
            button = ttk.Radiobutton(
                rotation_grid,
                text=f"{value * 60}°",
                variable=self.rotation,
                value=value,
                command=self._update_selected_preview,
            )
            button.grid(row=value // 2, column=value % 2, sticky="w", padx=(0, 8), pady=2)

        ttk.Label(frame, text="当前笔刷").grid(row=13, column=0, sticky="w", pady=(12, 3))
        self.preview_label = ttk.Label(frame, text="未选择", anchor=tk.CENTER)
        self.preview_label.grid(row=14, column=0, sticky="ew", pady=(0, 5))
        self.selected_path_label = ttk.Label(frame, text="", wraplength=210, justify=tk.LEFT)
        self.selected_path_label.grid(row=15, column=0, sticky="ew")

        ttk.Separator(frame).grid(row=16, column=0, sticky="ew", pady=10)
        ttk.Button(frame, text="打开球面拓扑检查器", command=self.open_topology_viewer).grid(
            row=17, column=0, sticky="ew"
        )

        ttk.Button(frame, text="打开分块与 Pack 检查器", command=self.open_chunk_storage_viewer).grid(
            row=18, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Button(frame, text="打开球面地图编辑器", command=self.open_sphere_editor).grid(
            row=19, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Button(frame, text="打开千万格生产构建器", command=self.open_production_builder).grid(
            row=20, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Button(frame, text="打开千万格生产 GPU 编辑器", command=self.open_production_editor).grid(
            row=21, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Button(frame, text="运行最终验收与 Windows 诊断", command=self.open_acceptance_viewer).grid(
            row=22, column=0, sticky="ew", pady=(6, 0)
        )

        ttk.Separator(frame).grid(row=23, column=0, sticky="ew", pady=10)
        ttk.Label(
            frame,
            text=(
                "当前版本已实装：笔刷与局部编辑、稳定球面拓扑、连通分块、层级区块可见性索引、"
                "Pack 增量保存与安全整理、球面编辑、后台区块读取、四级笔刷 LOD、区块远景缩略图、星球远景缓存、"
                "GPU 实例批次、2D 纹理数组、着色器六方向 UV 旋转、frequency=1004 程序化拓扑与紧凑分块索引。\n\n"
                "当前生产编辑器已接入 f1004 可见性层级、后台 Pack 读取、LOD0～LOD4、"
                "多层聚合远景、可见 GPU 实例流、纹理槽引用回收与分帧加载。"
                "程序可自动生成最终验收报告；真实 Windows/WGL 显卡结果必须在目标电脑运行诊断后确认。"
            ),
            wraplength=220,
            justify=tk.LEFT,
        ).grid(row=24, column=0, sticky="w")

    def open_topology_viewer(self) -> None:
        TopologyViewer(self.root, self.paths)

    def open_chunk_storage_viewer(self) -> None:
        ChunkStorageViewer(self.root, self.paths)

    def open_sphere_editor(self) -> None:
        SphereMapEditor(self.root, self.paths)

    def open_production_builder(self) -> None:
        ProductionBuilder(self.root, self.paths)

    def open_production_editor(self) -> None:
        ProductionSphereEditor(self.root, self.paths)

    def open_acceptance_viewer(self) -> None:
        AcceptanceViewer(self.root, self.paths)

    def refresh_brushes(self, show_dialog: bool = True) -> None:
        try:
            self.scan_result = self.catalog.scan()
        except Exception as exc:
            messagebox.showerror("错误", f"刷新笔刷库失败：\n{exc}")
            return
        self.records_by_uid = {record.uid: record for record in self.scan_result.records}
        self.thumbnail_cache.clear()
        self._populate_brush_tree()
        self._show_issues()
        self._redraw_map()
        self.status_text.set(
            f"笔刷扫描完成：有效 {self.scan_result.active_count}，缺失 {self.scan_result.missing_count}，无效 {len(self.scan_result.invalid)}"
        )
        if show_dialog:
            messagebox.showinfo("笔刷库", self.status_text.get())

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
            display_name = Path(record.relative_path).name
            self.brush_tree.insert(parent, tk.END, text=display_name, values=(record.uid,), tags=(f"uid:{record.uid}",))

    def _show_issues(self) -> None:
        lines: list[str] = []
        if self.scan_result is not None:
            for invalid in self.scan_result.invalid:
                lines.append(f"无效：{invalid.relative_path}\n  {self._reason_text(invalid.reason)}")
            for record in self.scan_result.records:
                if record.state == "missing":
                    lines.append(f"缺失：{record.relative_path}\n  UID 保留，地图引用未清除")
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

    def _update_selected_preview(self) -> None:
        if self.selected_brush_uid is None:
            self.preview_label.configure(image="", text="未选择")
            self.selected_path_label.configure(text="")
            return
        record = self.records_by_uid.get(self.selected_brush_uid)
        if record is None or record.state != "active":
            image = self.thumbnail_cache.get_missing(self.rotation.get())
            self.preview_label.configure(image=image, text="")
            self.preview_label.image = image
            self.selected_path_label.configure(text="笔刷文件缺失")
            return
        image = self.thumbnail_cache.get(record.uid, record.relative_path, self.rotation.get())
        self.preview_label.configure(image=image, text="")
        self.preview_label.image = image
        self.selected_path_label.configure(text=record.relative_path)

    def _calculate_centers(self) -> list[tuple[float, float]]:
        centers: list[tuple[float, float]] = []
        horizontal_step = self.side * 1.5
        for row in range(self.current_map.rows):
            for column in range(self.current_map.columns):
                center_x = self.margin + self.side + column * horizontal_step
                center_y = self.margin + self.hex_height / 2 + row * self.hex_height
                if column % 2 == 1:
                    center_y += self.hex_height / 2
                centers.append((center_x, center_y))
        return centers

    def _hex_points(self, center_x: float, center_y: float) -> list[float]:
        half_height = self.hex_height / 2
        return [
            center_x + self.side,
            center_y,
            center_x + self.side / 2,
            center_y + half_height,
            center_x - self.side / 2,
            center_y + half_height,
            center_x - self.side,
            center_y,
            center_x - self.side / 2,
            center_y - half_height,
            center_x + self.side / 2,
            center_y - half_height,
        ]

    def _redraw_map(self) -> None:
        self.canvas.delete("all")
        self.cell_centers = self._calculate_centers()
        for index, (center_x, center_y) in enumerate(self.cell_centers):
            uid, rotation = self.current_map.brush_uid_for_cell(index)
            tags = (f"cell:{index}", "cell")
            if uid is not None:
                record = self.records_by_uid.get(uid)
                if record is not None and record.state == "active":
                    image = self.thumbnail_cache.get(uid, record.relative_path, rotation)
                else:
                    image = self.thumbnail_cache.get_missing(rotation)
                self.canvas.create_image(center_x, center_y, image=image, tags=tags)
            self.canvas.create_polygon(
                self._hex_points(center_x, center_y),
                fill="" if uid is not None else "#343a40",
                outline="#89929b",
                width=1,
                tags=tags,
            )
        if self.cell_centers:
            max_x = max(center[0] for center in self.cell_centers) + self.side + self.margin
            max_y = max(center[1] for center in self.cell_centers) + self.hex_height / 2 + self.margin
            self.canvas.configure(scrollregion=(0, 0, max_x, max_y))

    def _on_canvas_click(self, event: tk.Event) -> None:
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        index = self._cell_index_at(canvas_x, canvas_y)
        if index is None:
            return
        if self.selected_tool.get() == "erase":
            self.current_map.set_cell(index, None)
        else:
            if self.selected_brush_uid is None:
                self.status_text.set("请先选择一张有效笔刷")
                return
            record = self.records_by_uid.get(self.selected_brush_uid)
            if record is None or record.state != "active":
                self.status_text.set("当前笔刷文件缺失，不能放置")
                return
            self.current_map.set_cell(index, record.uid, record.relative_path, self.rotation.get())
        self._redraw_map()
        self.status_text.set(f"已修改格子 {index}，地图尚未保存")

    def _cell_index_at(self, x: float, y: float) -> int | None:
        half_height = self.hex_height / 2
        for index, (center_x, center_y) in enumerate(self.cell_centers):
            dx = abs(x - center_x)
            dy = abs(y - center_y)
            if dx > self.side or dy > half_height:
                continue
            if math.sqrt(3) * dx + dy <= math.sqrt(3) * self.side + 0.5:
                return index
        return None

    def new_map(self) -> None:
        if not self._confirm_discard_changes():
            return
        try:
            self.current_map = self.map_store.create_blank(self.map_name.get())
        except ValueError as exc:
            messagebox.showerror("地图名称无效", str(exc))
            return
        self.map_name.set(self.current_map.name)
        self._redraw_map()
        self.status_text.set(f"已新建局部地图：{self.current_map.name}")

    def save_map(self) -> None:
        try:
            self.current_map.name = self.map_name.get().strip()
            map_dir = self.map_store.save(self.current_map)
        except (ValueError, MapFormatError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.map_name.set(self.current_map.name)
        self._refresh_map_choices()
        self.status_text.set(f"地图已保存：{map_dir}")

    def open_map(self) -> None:
        name = self.map_choice.get().strip()
        if not name:
            messagebox.showinfo("打开地图", "没有可打开的地图")
            return
        if not self._confirm_discard_changes():
            return
        try:
            self.current_map = self.map_store.load(name)
        except (ValueError, MapFormatError, OSError) as exc:
            messagebox.showerror("打开失败", str(exc))
            return
        self.map_name.set(self.current_map.name)
        self._redraw_map()
        self.status_text.set(f"已打开地图：{name}")

    def _refresh_map_choices(self) -> None:
        names = self.map_store.list_maps()
        self.map_combo.configure(values=names)
        if names:
            if self.map_choice.get() not in names:
                self.map_choice.set(names[0])
        else:
            self.map_choice.set("")

    def _confirm_discard_changes(self) -> bool:
        if not self.current_map.dirty:
            return True
        return messagebox.askyesno("尚未保存", "当前地图有未保存修改，确定继续吗？")

    def _on_close(self) -> None:
        if self._confirm_discard_changes():
            self.root.destroy()
