from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .distant_lod import DistantLodCache, EMPTY_RGB, MISSING_RGB
from .flat_map import FlatCellPolygon, IcosahedralNetLayout, render_net_surface_ppm
from .software_globe import point_in_polygon


class ProductionFlatMapEditor:
    """2D icosahedron-net editor sharing the host's authoritative map session."""

    def __init__(self, host) -> None:
        self.host = host
        if host.topology is None or host.layout is None or host.session is None:
            raise RuntimeError("完整星球地图尚未准备完成")
        self.window = tk.Toplevel(host.window)
        self.window.title("完整星球 2D 展开编辑器 v1.3.1")
        self.window.geometry("1280x820")
        self.window.minsize(980, 650)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Control-s>", lambda _event: self.host.save())
        self.window.bind("<Control-z>", lambda _event: self.host.undo())
        self.window.bind("<Control-y>", lambda _event: self.host.redo())
        self.window.bind("<Control-Shift-Z>", lambda _event: self.host.redo())

        self.net = IcosahedralNetLayout(host.topology.frequency)
        left, top, right, bottom = self.net.bounds
        self.center_x = (left + right) / 2.0
        self.center_y = (top + bottom) / 2.0
        self.view_scale = 0.18
        self.pan_origin: tuple[int, int, float, float] | None = None
        self.panning = False
        self.visible_polygons: tuple[tuple[FlatCellPolygon, tuple[float, ...]], ...] = ()
        self.cursor_position: tuple[int, int] | None = None
        self.stroke_id: int | None = None
        self.stroke_tool = None
        self.last_stroke_cell: int | None = None
        self.render_generation = 0
        self.render_running = False
        self.pending_render: tuple[object, ...] | None = None
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.photo: tk.PhotoImage | None = None
        self.color_cache = DistantLodCache(host.paths.brush_root)
        self.last_dirty_signature: tuple[int, ...] = ()
        self.last_surface_signature: str | None = None
        self.status = tk.StringVar(value="2D展开视图与球面视图共用同一张地图")
        self.detail_info = tk.StringVar(value="滚轮缩放；右键拖动画布；任意缩放下按住左键连续绘制")

        self._build_ui()
        self.window.after(40, self._poll)
        self.window.after(80, self._fit_view)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(4, weight=1)
        ttk.Button(toolbar, text="显示完整展开图", command=self._fit_view).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(toolbar, text="保存星球地图", command=self.host.save).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(toolbar, text="回到球面窗口", command=self._raise_host).grid(row=0, column=2, padx=(0, 10))
        ttk.Label(toolbar, textvariable=self.detail_info).grid(row=0, column=4, sticky="e")

        frame = ttk.LabelFrame(
            root,
            text="二十面体展开图（接缝格子会重复显示，但都引用同一个 CellId）",
            padding=4,
        )
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(frame, background="#14191f", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._request_draw())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.16))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.16))
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)
        self.canvas.bind("<ButtonPress-1>", self._paint_start)
        self.canvas.bind("<B1-Motion>", self._paint_move)
        self.canvas.bind("<ButtonRelease-1>", self._paint_end)
        self.canvas.bind("<Motion>", self._track_cursor)
        self.canvas.bind("<Leave>", self._clear_cursor)
        self.brush_cursor_item = self.canvas.create_oval(
            0, 0, 0, 0, fill="", outline="#ffd166", width=2,
            state=tk.HIDDEN, tags=("cursor",),
        )

        ttk.Label(
            self.window,
            textvariable=self.status,
            anchor=tk.W,
            relief=tk.SUNKEN,
            padding=(8, 4),
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _raise_host(self) -> None:
        try:
            self.host.window.lift()
            self.host.window.focus_force()
        except tk.TclError:
            pass

    def _fit_view(self) -> None:
        width = max(400, self.canvas.winfo_width() - 40)
        height = max(300, self.canvas.winfo_height() - 40)
        left, top, right, bottom = self.net.bounds
        self.center_x = (left + right) / 2.0
        self.center_y = (top + bottom) / 2.0
        self.view_scale = min(width / (right - left), height / (bottom - top))
        self._request_draw()

    def _world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            (x - self.center_x) * self.view_scale + width / 2.0,
            (y - self.center_y) * self.view_scale + height / 2.0,
        )

    def _canvas_to_world(self, x: float, y: float) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            self.center_x + (x - width / 2.0) / self.view_scale,
            self.center_y + (y - height / 2.0) / self.view_scale,
        )

    def _world_bounds(self) -> tuple[float, float, float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        half_width = width / (2.0 * self.view_scale)
        half_height = height / (2.0 * self.view_scale)
        return (
            self.center_x - half_width,
            self.center_y - half_height,
            self.center_x + half_width,
            self.center_y + half_height,
        )

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1.16 if event.delta > 0 else 1 / 1.16)

    def _zoom_at(self, canvas_x: int, canvas_y: int, factor: float) -> None:
        before = self._canvas_to_world(canvas_x, canvas_y)
        self.view_scale = max(0.03, min(96.0, self.view_scale * factor))
        after = self._canvas_to_world(canvas_x, canvas_y)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]
        self._request_draw(preview=True)
        self._update_brush_cursor()

    def _pan_start(self, event: tk.Event) -> None:
        self.pan_origin = (event.x, event.y, self.center_x, self.center_y)
        self.panning = True

    def _pan_move(self, event: tk.Event) -> None:
        if self.pan_origin is None:
            return
        start_x, start_y, center_x, center_y = self.pan_origin
        self.center_x = center_x - (event.x - start_x) / self.view_scale
        self.center_y = center_y - (event.y - start_y) / self.view_scale
        self._request_draw(preview=True)

    def _pan_end(self, _event: tk.Event) -> None:
        self.pan_origin = None
        self.panning = False
        self._request_draw()

    def _request_draw(self, *, preview: bool = False) -> None:
        self.render_generation += 1
        generation = self.render_generation
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        texture = self.host.surface_texture
        self.pending_render = (
            generation,
            texture,
            self.center_x,
            self.center_y,
            self.view_scale,
            width,
            height,
            4 if preview or self.panning else 2,
        )
        if not self.render_running:
            self._start_render()
        self._draw_detail_overlay()

    def _start_render(self) -> None:
        request = self.pending_render
        if request is None or self.render_running:
            return
        self.pending_render = None
        generation, texture, center_x, center_y, scale, width, height, block = request
        if texture is None:
            self.canvas.delete("background")
            self._draw_net_lines()
            return
        self.render_running = True

        def worker() -> None:
            try:
                ppm = render_net_surface_ppm(
                    texture,
                    self.net,
                    center_x,
                    center_y,
                    scale,
                    width,
                    height,
                    block_size=block,
                )
                self.results.put(("render", (generation, ppm)))
            except Exception as exc:
                self.results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-flat-render", daemon=True).start()

    def _draw_net_lines(self) -> None:
        self.canvas.delete("net")
        for placement in self.net.face_placements:
            points = tuple(value for point in placement.vertex_points for value in self._world_to_canvas(*point))
            self.canvas.create_polygon(points, fill="", outline="#b8c7cf", width=1, tags=("net",))
            if self.view_scale >= 0.35:
                cx = sum(point[0] for point in placement.vertex_points) / 3.0
                cy = sum(point[1] for point in placement.vertex_points) / 3.0
                sx, sy = self._world_to_canvas(cx, cy)
                self.canvas.create_text(sx, sy, text=str(placement.face_id), fill="#dce6eb", tags=("net",))
        self.canvas.tag_raise("net")
        self.canvas.tag_raise("cursor")

    def _draw_detail_overlay(self) -> None:
        self.canvas.delete("detail")
        self.visible_polygons = ()
        topology = self.host.topology
        layout = self.host.layout
        session = self.host.session
        if topology is None or layout is None or session is None:
            return
        if self.view_scale < 9.0:
            self.detail_info.set(
                f"缩放 {self.view_scale:.2f}px/格；可直接绘制，"
                "放大到约9px/格后才逐格显示（远景会实时更新）"
            )
            self._draw_net_lines()
            return
        cells = self.net.visible_cells(topology, self._world_bounds(), maximum_cells=8000)
        if not cells:
            self.detail_info.set("当前范围格子过多；仍可绘制，放大后才逐格显示")
            self._draw_net_lines()
            return

        active_chunks: set[int] = set()
        values_by_cell: dict[int, int] = {}
        for cell in cells:
            if cell.cell_id in values_by_cell:
                continue
            chunk_id, local_index = layout.chunk_for_cell(cell.cell_id)
            active_chunks.add(chunk_id)
            values = self.host.store.load_chunk(session, chunk_id)
            values_by_cell[cell.cell_id] = values[local_index]
        for chunk_id in tuple(session.loaded_chunks):
            if chunk_id not in active_chunks and chunk_id not in session.dirty_chunks:
                session.loaded_chunks.pop(chunk_id, None)

        rendered: list[tuple[FlatCellPolygon, tuple[float, ...]]] = []
        for cell in cells:
            canvas_points: list[float] = []
            for index in range(0, len(cell.points), 2):
                sx, sy = self._world_to_canvas(cell.points[index], cell.points[index + 1])
                canvas_points.extend((sx, sy))
            value = values_by_cell[cell.cell_id]
            local_id = value & 0x0FFF
            color = EMPTY_RGB
            if local_id:
                entry = session.brush_entries.get(local_id)
                record = None if entry is None else self.host.records_by_uid.get(entry.brush_uid)
                color = self.color_cache.representative_color(record) if record is not None else MISSING_RGB
            fill = "#%02x%02x%02x" % color
            edge = "#%02x%02x%02x" % tuple(max(0, int(channel * 0.55)) for channel in color)
            flat = tuple(canvas_points)
            self.canvas.create_polygon(flat, fill=fill, outline=edge, width=1, tags=("detail",))
            rendered.append((cell, flat))
        self.visible_polygons = tuple(rendered)
        self._draw_net_lines()
        self.canvas.tag_raise("detail")
        self.canvas.tag_raise("net")
        self.detail_info.set(
            f"详细格子 {len(rendered):,}；左键连续绘制；当前笔刷直径 {self.host._brush_diameter_value()} 格"
        )

    def _track_cursor(self, event: tk.Event) -> None:
        self.cursor_position = (int(event.x), int(event.y))
        self._update_brush_cursor()

    def _clear_cursor(self, _event: tk.Event | None = None) -> None:
        self.cursor_position = None
        self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)

    def _update_brush_cursor(self) -> None:
        """Show the brush footprint in net space.

        One world unit on the net is one cell, so the footprint is simply the
        graph radius scaled by the current zoom.
        """
        position = self.cursor_position
        if position is None:
            self.canvas.itemconfigure(self.brush_cursor_item, state=tk.HIDDEN)
            return
        radius = (self.host._brush_diameter_value() / 2.0 + 0.5) * self.view_scale
        if radius < 1.0:
            radius = 1.0
        x, y = position
        outline = {"paint": "#ffd166", "erase": "#ff8f8f", "undo": "#8fd0ff"}.get(
            self.host.tool.get(), "#ffd166"
        )
        self.canvas.coords(
            self.brush_cursor_item, x - radius, y - radius, x + radius, y + radius
        )
        self.canvas.itemconfigure(
            self.brush_cursor_item, state=tk.NORMAL, outline=outline
        )
        self.canvas.tag_raise(self.brush_cursor_item)

    def refresh_after_external_edit(self) -> None:
        """Redraw after the sphere editor changed shared cells (undo/redo)."""
        try:
            self._request_draw()
        except tk.TclError:
            pass

    def _hit_cell(self, x: float, y: float) -> int | None:
        for cell, points in reversed(self.visible_polygons):
            if point_in_polygon(x, y, points):
                return cell.cell_id
        world_x, world_y = self._canvas_to_world(x, y)
        hit = self.net.nearest_cell(self.host.topology, world_x, world_y)
        return None if hit is None else hit[1]

    def _paint_start(self, event: tk.Event) -> None:
        # No zoom gate: ``_hit_cell`` falls back to an exact barycentric lookup on
        # the net, which resolves a single CellId at any scale.  Painting from the
        # overview simply means the result is only visible after the next save.
        cell_id = self._hit_cell(event.x, event.y)
        if cell_id is None:
            return
        try:
            tool = self.host._stroke_tool_from_ui()
        except Exception as exc:
            messagebox.showerror("2D展开编辑器", str(exc), parent=self.window)
            return
        self.host.local_stroke_id += 1
        self.stroke_id = -self.host.local_stroke_id
        self.stroke_tool = tool
        self.last_stroke_cell = cell_id
        self.host._queue_stroke_segment("flat", self.stroke_id, "start", cell_id, tool)
        self.status.set(f"开始2D连续绘制；直径 {tool.diameter} 格")

    def _paint_move(self, event: tk.Event) -> None:
        self._track_cursor(event)
        if self.stroke_id is None or self.stroke_tool is None:
            return
        cell_id = self._hit_cell(event.x, event.y)
        if cell_id is None or cell_id == self.last_stroke_cell:
            return
        self.last_stroke_cell = cell_id
        self.host._queue_stroke_segment("flat", self.stroke_id, "move", cell_id, self.stroke_tool)

    def _paint_end(self, _event: tk.Event | None = None) -> None:
        if self.stroke_id is None:
            return
        stroke_id = self.stroke_id
        tool = self.stroke_tool
        self.stroke_id = None
        self.stroke_tool = None
        self.last_stroke_cell = None
        self.host._queue_stroke_segment("flat", stroke_id, "end", -1, tool)
        self.status.set("2D笔划已结束；按 Ctrl+S 保存")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "render":
                    self.render_running = False
                    generation, ppm = payload
                    if generation == self.render_generation:
                        self.photo = tk.PhotoImage(master=self.window, data=ppm, format="PPM")
                        self.canvas.delete("background")
                        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags=("background",))
                        self.canvas.tag_lower("background")
                        self._draw_detail_overlay()
                    self._start_render()
                elif kind == "error":
                    self.render_running = False
                    self.status.set(f"2D地表渲染失败：{payload}")
                    self._start_render()
        except queue.Empty:
            pass

        session = self.host.session
        dirty_signature = () if session is None else tuple(sorted(session.dirty_chunks))
        surface_signature = self.host.surface_signature
        if dirty_signature != self.last_dirty_signature:
            self.last_dirty_signature = dirty_signature
            self._draw_detail_overlay()
        if surface_signature != self.last_surface_signature:
            self.last_surface_signature = surface_signature
            self._request_draw()
        if self.window.winfo_exists():
            self.window.after(80, self._poll)

    def close(self) -> None:
        if self.stroke_id is not None:
            self._paint_end()
        try:
            self.host._flat_editors.discard(self)
        except Exception:
            pass
        try:
            self.window.destroy()
        except tk.TclError:
            pass
