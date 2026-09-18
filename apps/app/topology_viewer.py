from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .paths import ProjectPaths
from .topology import DualTopology, TopologyError, generate_dual_topology, write_topology_cache
from .i18n import t


class TopologyViewer:
    def __init__(self, parent: tk.Misc, paths: ProjectPaths) -> None:
        self.paths = paths
        self.window = tk.Toplevel(parent)
        self.window.title(t("球面拓扑检查器 v1.3.1"))
        self.window.geometry("1120x780")
        self.window.minsize(820, 600)

        self.frequency = tk.StringVar(value="8")
        self.status = tk.StringVar(value=t("选择频率后生成测试拓扑"))
        self.summary = tk.StringVar(value=t("尚未生成"))
        self.selected = tk.StringVar(value=t("未选择格子"))
        self.topology: DualTopology | None = None
        self.yaw = -0.35
        self.pitch = 0.25
        self.drag_origin: tuple[int, int, float, float] | None = None
        self.projected_cells: list[tuple[float, float, int]] = []
        self.generating = False
        self.generation_results: queue.Queue[tuple[str, object, object | None]] = queue.Queue()

        self._build_ui()
        self.window.after(100, self.generate)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        viewer_frame = ttk.LabelFrame(root, text=t("对偶球面格子预览"), padding=6)
        viewer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        viewer_frame.rowconfigure(0, weight=1)
        viewer_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(viewer_frame, background="#171b20", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Button-3>", self._select_cell)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        panel = ttk.LabelFrame(root, text=t("生成与验证"), padding=10)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text=t("测试细分频率")).grid(row=0, column=0, sticky="w")
        self.frequency_combo = ttk.Combobox(
            panel,
            textvariable=self.frequency,
            values=("1", "2", "4", "8", "16", "32"),
            state="readonly",
            width=20,
        )
        self.frequency_combo.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.generate_button = ttk.Button(panel, text=t("生成并写入拓扑缓存"), command=self.generate)
        self.generate_button.grid(row=2, column=0, sticky="ew")

        ttk.Separator(panel).grid(row=3, column=0, sticky="ew", pady=10)
        ttk.Label(panel, text=t("验证结果")).grid(row=4, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.summary, wraplength=250, justify=tk.LEFT).grid(
            row=5, column=0, sticky="ew", pady=(3, 8)
        )
        ttk.Label(panel, textvariable=self.selected, wraplength=250, justify=tk.LEFT).grid(
            row=6, column=0, sticky="ew", pady=(0, 8)
        )

        ttk.Separator(panel).grid(row=7, column=0, sticky="ew", pady=10)
        ttk.Label(
            panel,
            text=(
                t("左键拖动：旋转球体\n"
                "右键点击：查看最近格子\n\n"
                "浅色轮廓为普通六边形格子，橙色区域为拓扑中必须存在的 12 个五边形。")
            ),
            wraplength=250,
            justify=tk.LEFT,
        ).grid(row=8, column=0, sticky="w")

        ttk.Separator(panel).grid(row=9, column=0, sticky="ew", pady=10)
        ttk.Label(
            panel,
            text=(
                t("本检查器使用完整对偶拓扑数据，但界面只开放到 frequency=32，避免 Python 原型因一次生成过大数据而长时间占用内存。\n\n"
                "正式目标 frequency=1004 的 CellId 公式和数量规则已经与此生成器共用，但本版本尚未声称完成千万格生产缓存。")
            ),
            wraplength=250,
            justify=tk.LEFT,
        ).grid(row=10, column=0, sticky="w")

        status = ttk.Label(self.window, textvariable=self.status, anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def generate(self) -> None:
        if self.generating:
            return
        try:
            frequency = int(self.frequency.get())
        except ValueError:
            messagebox.showerror(t("错误"), t("细分频率无效"), parent=self.window)
            return

        self.generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.status.set(t("正在生成 frequency={frequency} 的球面拓扑……", frequency=frequency))
        worker = threading.Thread(target=self._generate_worker, args=(frequency,), daemon=True)
        worker.start()
        self.window.after(50, self._poll_generation)

    def _generate_worker(self, frequency: int) -> None:
        try:
            topology = generate_dual_topology(frequency, max_cells=10 * 32 * 32 + 2)
            cache_info = write_topology_cache(topology, self.paths.map_root)
        except Exception as exc:
            self.generation_results.put(("error", exc, None))
            return
        self.generation_results.put(("success", topology, cache_info.directory))

    def _poll_generation(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            state, value, extra = self.generation_results.get_nowait()
        except queue.Empty:
            if self.generating:
                self.window.after(50, self._poll_generation)
            return
        if state == "error":
            self._generation_failed(value if isinstance(value, Exception) else TopologyError(str(value)))
            return
        if not isinstance(value, DualTopology):
            self._generation_failed(TopologyError("Topology worker returned an invalid result"))
            return
        self._generation_finished(value, extra)

    def _generation_failed(self, error: Exception) -> None:
        self.generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.status.set(t("拓扑生成失败"))
        messagebox.showerror(t("拓扑生成失败"), str(error), parent=self.window)

    def _generation_finished(self, topology: DualTopology, directory) -> None:
        self.generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.topology = topology
        validation = topology.validate()
        self.summary.set(
            "\n".join(
                (
                    t("Cell：{cell_count:,}", cell_count=validation.cell_count),
                    t("三角面：{triangle_count:,}", triangle_count=validation.triangle_count),
                    t("拓扑边：{edge_count:,}", edge_count=validation.edge_count),
                    t("五边形：{pentagon_count}", pentagon_count=validation.pentagon_count),
                    t("六边形：{hexagon_count:,}", hexagon_count=validation.hexagon_count),
                    t("Euler：{euler_characteristic}", euler_characteristic=validation.euler_characteristic),
                    t("邻接互反：{value}", value=t('通过') if validation.reciprocal_neighbor_links else t('失败')),
                    t("稳定哈希：{stable_hash}…", stable_hash=validation.stable_hash[:16]),
                )
            )
        )
        self.status.set(t("拓扑已生成并验证，缓存：{directory}", directory=directory))
        self.selected.set(t("未选择格子"))
        self._redraw()

    @staticmethod
    def _rotate(
        point: tuple[float, float, float], yaw: float, pitch: float
    ) -> tuple[float, float, float]:
        x, y, z = point
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x, z = x * cos_yaw + z * sin_yaw, -x * sin_yaw + z * cos_yaw
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        y, z = y * cos_pitch - z * sin_pitch, y * sin_pitch + z * cos_pitch
        return x, y, z

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self.projected_cells.clear()
        topology = self.topology
        if topology is None:
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=t("正在准备球面拓扑……"),
                fill="#c8d0d8",
                font=("TkDefaultFont", 13),
            )
            return

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        radius = max(20.0, min(width, height) * 0.43)
        center_x = width / 2.0
        center_y = height / 2.0
        rotated_cells = [self._rotate(point, self.yaw, self.pitch) for point in topology.cell_centers]
        rotated_corners = [self._rotate(point, self.yaw, self.pitch) for point in topology.triangle_centers]
        pentagon_set = set(topology.pentagon_ids)

        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill="#222a31",
            outline="#687480",
            width=2,
        )

        draw_order = sorted(
            (cell_id for cell_id, point in enumerate(rotated_cells) if point[2] > -0.03),
            key=lambda cell_id: rotated_cells[cell_id][2],
        )
        for cell_id in draw_order:
            corner_ids = topology.incident_triangles[cell_id]
            points: list[float] = []
            visible_corners = 0
            for corner_id in corner_ids:
                x, y, z = rotated_corners[corner_id]
                points.extend((center_x + x * radius, center_y - y * radius))
                if z >= -0.04:
                    visible_corners += 1
            if visible_corners < len(corner_ids) - 1:
                continue
            depth = max(0.0, min(1.0, rotated_cells[cell_id][2]))
            outline = self._shade("#8b98a4", depth)
            fill = "#a86635" if cell_id in pentagon_set else ""
            self.canvas.create_polygon(points, fill=fill, outline=outline, width=1)

        for cell_id, (x, y, z) in enumerate(rotated_cells):
            if z <= 0.0:
                continue
            screen_x = center_x + x * radius
            screen_y = center_y - y * radius
            self.projected_cells.append((screen_x, screen_y, cell_id))

    @staticmethod
    def _shade(base: str, depth: float) -> str:
        red = int(base[1:3], 16)
        green = int(base[3:5], 16)
        blue = int(base[5:7], 16)
        factor = 0.45 + 0.55 * depth
        return f"#{int(red * factor):02x}{int(green * factor):02x}{int(blue * factor):02x}"

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_origin = (event.x, event.y, self.yaw, self.pitch)

    def _drag(self, event: tk.Event) -> None:
        if self.drag_origin is None:
            return
        start_x, start_y, start_yaw, start_pitch = self.drag_origin
        self.yaw = start_yaw + (event.x - start_x) * 0.008
        self.pitch = max(-1.45, min(1.45, start_pitch + (event.y - start_y) * 0.008))
        self._redraw()

    def _end_drag(self, _event: tk.Event) -> None:
        self.drag_origin = None

    def _select_cell(self, event: tk.Event) -> None:
        topology = self.topology
        if topology is None or not self.projected_cells:
            return
        nearest = min(
            self.projected_cells,
            key=lambda item: (item[0] - event.x) ** 2 + (item[1] - event.y) ** 2,
        )
        distance = math.sqrt((nearest[0] - event.x) ** 2 + (nearest[1] - event.y) ** 2)
        if distance > 24:
            self.selected.set(t("未选择格子"))
            return
        cell_id = nearest[2]
        is_pentagon = cell_id in set(topology.pentagon_ids)
        self.selected.set(
            "\n".join(
                (
                    t("CellId：{cell_id}", cell_id=cell_id),
                    t("类型：{value}", value=t('隐藏五边形') if is_pentagon else t('普通六边形')),
                    t("邻居数：{len}", len=len(topology.neighbors[cell_id])),
                    t("邻居：") + ", ".join(str(value) for value in topology.neighbors[cell_id]),
                )
            )
        )
