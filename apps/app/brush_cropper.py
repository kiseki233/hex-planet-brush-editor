from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .brush_cropper_core import (
    TILE_SIZE,
    BrushCropError,
    SourceTransform,
    TileSelection,
    export_selection,
    selection_states,
    source_world_bounds,
    visible_grid_range,
)
from .paths import ProjectPaths
from .png_pixels import PixelImage, PngPixelError, read_png_pixels
from .i18n import t


class BrushImageCropper:
    VIEW_PERCENTAGES = (12.5, 25.0, 50.0, 75.0, 100.0)

    def __init__(
        self,
        parent: tk.Misc,
        paths: ProjectPaths,
    ) -> None:
        self.paths = paths
        self.window = tk.Toplevel(parent)
        self.window.title(t("地图图片裁剪器 v1.3.1"))
        self.window.geometry("1220x820")
        self.window.minsize(980, 680)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self.source_path: Path | None = None
        self.source: PixelImage | None = None
        self.transform = SourceTransform()
        self.selection = TileSelection(0, 0, 0, 0)
        self.selection_start: tuple[int, int] | None = None
        self.source_drag_origin: tuple[int, int, SourceTransform] | None = None
        self.view_drag_origin: tuple[int, int, float, float] | None = None
        self.view_center_x = TILE_SIZE / 2
        self.view_center_y = TILE_SIZE / 2
        self.view_scale = 0.5
        self.render_generation = 0
        self.render_running = False
        self.pending_render: tuple[int, int, int, bool] | None = None
        self.render_results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.render_job: str | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.export_running = False

        self.source_info = tk.StringVar(value=t("尚未打开源图"))
        self.image_scale_percent = tk.DoubleVar(value=100.0)
        self.view_percent = tk.StringVar(value="50%")
        self.next_output_name = tk.StringVar(value=t("下一张：001.png"))
        self.selection_info = tk.StringVar(value=t("已选择 1×1，共1张"))
        self.status = tk.StringVar(value=t("先打开源图；导入后不会自动裁剪"))

        self._build_ui()
        self._update_next_output_name()
        self.window.after(40, self._poll_results)
        self._request_render()

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.LabelFrame(root, text=t("源图与比例"), padding=8)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(4, weight=1)

        ttk.Button(toolbar, text=t("打开源图 PNG"), command=self._choose_source).grid(
            row=0, column=0, rowspan=2, sticky="ns", padx=(0, 8)
        )
        ttk.Label(toolbar, text=t("图片实际缩放")).grid(row=0, column=1, sticky="w")
        scale_spin = ttk.Spinbox(
            toolbar,
            from_=1.0,
            to=3200.0,
            increment=1.0,
            textvariable=self.image_scale_percent,
            width=9,
            command=self._scale_entry_changed,
        )
        scale_spin.grid(row=1, column=1, sticky="w")
        scale_spin.bind("<Return>", self._scale_entry_changed)
        scale_spin.bind("<FocusOut>", self._scale_entry_changed)
        ttk.Label(toolbar, text="%").grid(row=1, column=2, sticky="w", padx=(2, 12))

        ttk.Button(toolbar, text="100%", command=self._reset_image_scale).grid(
            row=0, column=3, rowspan=2, sticky="ns", padx=(0, 10)
        )
        ttk.Label(toolbar, textvariable=self.source_info, wraplength=480).grid(
            row=0, column=4, rowspan=2, sticky="w"
        )
        ttk.Label(toolbar, text=t("工作区查看倍率")).grid(row=0, column=5, sticky="w", padx=(10, 0))
        view_combo = ttk.Combobox(
            toolbar,
            state="readonly",
            textvariable=self.view_percent,
            values=tuple(f"{value:g}%" for value in self.VIEW_PERCENTAGES),
            width=8,
        )
        view_combo.grid(row=1, column=5, sticky="w", padx=(10, 0))
        view_combo.bind("<<ComboboxSelected>>", self._view_scale_changed)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(
            body,
            text=(
                t("裁剪工作区：滚轮缩放源图；右键拖动源图；中键拖动查看区域；"
                "左键拖动框选多个512×512格")
            ),
            padding=4,
        )
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#171b20", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._request_render())
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._scale_source_at(event.x, event.y, 1.1))
        self.canvas.bind("<Button-5>", lambda event: self._scale_source_at(event.x, event.y, 1 / 1.1))
        self.canvas.bind("<ButtonPress-3>", self._source_drag_start)
        self.canvas.bind("<B3-Motion>", self._source_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self._source_drag_end)
        self.canvas.bind("<ButtonPress-2>", self._view_drag_start)
        self.canvas.bind("<B2-Motion>", self._view_drag_move)
        self.canvas.bind("<ButtonRelease-2>", self._view_drag_end)
        self.canvas.bind("<ButtonPress-1>", self._selection_start)
        self.canvas.bind("<B1-Motion>", self._selection_move)
        self.canvas.bind("<ButtonRelease-1>", self._selection_end)

        panel = ttk.LabelFrame(body, text=t("裁剪与导出"), padding=10)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(0, weight=1)

        ttk.Label(
            panel,
            text=(
                t("每个网格输出一张512×512 PNG。六边形线表示地图中真正可见的区域，"
                "四角仍会保留在PNG中，但放入地图后不会显示。")
            ),
            wraplength=265,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")
        ttk.Separator(panel).grid(row=1, column=0, sticky="ew", pady=10)
        ttk.Label(panel, text=t("统一输出目录")).grid(row=2, column=0, sticky="w")
        ttk.Label(
            panel,
            text=t("art/data/（不建立分类子文件夹）"),
            wraplength=265,
        ).grid(row=3, column=0, sticky="w", pady=(2, 4))
        ttk.Label(panel, textvariable=self.next_output_name).grid(
            row=4, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(panel, textvariable=self.selection_info, wraplength=265).grid(
            row=6, column=0, sticky="w"
        )
        self.export_button = ttk.Button(
            panel,
            text=t("导出选中格子到 art/data"),
            command=self._export,
            state=tk.DISABLED,
        )
        self.export_button.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(panel, text=t("重新居中显示源图"), command=self._fit_source_view).grid(
            row=8, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Separator(panel).grid(row=9, column=0, sticky="ew", pady=10)
        ttk.Label(
            panel,
            text=(
                t("比例对齐方法：把源图上的3 km比例尺移动到画布上方的标尺下面，"
                "再用滚轮调整源图大小，直到两端长度一致。标尺宽度始终等于一个地图格。")
            ),
            wraplength=265,
            justify=tk.LEFT,
        ).grid(row=10, column=0, sticky="w")

        ttk.Label(self.window, textvariable=self.status, anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _choose_source(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title=t("选择需要裁剪的源图"),
            filetypes=((t("PNG图片"), "*.png"), (t("所有文件"), "*.*")),
        )
        if not path:
            return
        self.status.set(t("正在读取源图……"))
        self.export_button.configure(state=tk.DISABLED)

        def worker() -> None:
            try:
                image = read_png_pixels(path)
                self.render_results.put(("source", (Path(path), image)))
            except Exception as exc:
                self.render_results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-cropper-load", daemon=True).start()

    def load_source_path(self, path: str | Path) -> None:
        """Testable direct-load helper that bypasses the file dialog."""
        image = read_png_pixels(path)
        self._accept_source(Path(path), image)

    def _accept_source(self, path: Path, image: PixelImage) -> None:
        self.source_path = path
        self.source = image
        self.transform = SourceTransform(1.0, 0.0, 0.0)
        self.image_scale_percent.set(100.0)
        self.selection = TileSelection(0, 0, 0, 0)
        self.source_info.set(
            f"{path.name} · {image.width}×{image.height} · {'RGBA' if image.channels == 4 else 'RGB'}"
        )
        self._fit_source_view()
        self._update_selection_info()
        self.export_button.configure(state=tk.NORMAL)
        self.status.set(t("源图已载入；现在可以缩放、移动和框选，尚未生成任何裁剪文件"))

    def _fit_source_view(self) -> None:
        if self.source is None:
            return
        left, top, right, bottom = source_world_bounds(self.source, self.transform)
        self.view_center_x = (left + right) / 2.0
        self.view_center_y = (top + bottom) / 2.0
        width = max(200, self.canvas.winfo_width() - 80)
        height = max(200, self.canvas.winfo_height() - 80)
        fit_scale = min(width / max(1.0, right - left), height / max(1.0, bottom - top))
        options = [value / 100.0 for value in self.VIEW_PERCENTAGES]
        self.view_scale = max(options[0], min(options[-1], fit_scale))
        closest = min(options, key=lambda value: abs(value - self.view_scale))
        self.view_scale = closest
        self.view_percent.set(f"{closest * 100:g}%")
        self._request_render()

    def _reset_image_scale(self) -> None:
        if self.source is None:
            return
        world_x = self.view_center_x
        world_y = self.view_center_y
        self.transform = self.transform.scaled_about(1.0, world_x, world_y)
        self.image_scale_percent.set(100.0)
        self._request_render(preview=True)
        self._update_selection_info()

    def _scale_entry_changed(self, _event: tk.Event | None = None) -> None:
        if self.source is None:
            return
        try:
            percent = float(self.image_scale_percent.get())
        except (tk.TclError, ValueError):
            percent = self.transform.scale * 100.0
        percent = max(1.0, min(3200.0, percent))
        self.image_scale_percent.set(round(percent, 3))
        self.transform = self.transform.scaled_about(
            percent / 100.0,
            self.view_center_x,
            self.view_center_y,
        )
        self._request_render(preview=True)
        self._update_selection_info()

    def _view_scale_changed(self, _event: tk.Event | None = None) -> None:
        value = self.view_percent.get().rstrip("%")
        try:
            percent = float(value)
        except ValueError:
            percent = 50.0
        self.view_scale = max(0.05, min(2.0, percent / 100.0))
        self._request_render()

    def _mouse_wheel(self, event: tk.Event) -> None:
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self._scale_source_at(event.x, event.y, factor)

    def _scale_source_at(self, canvas_x: int, canvas_y: int, factor: float) -> None:
        if self.source is None:
            return
        world_x, world_y = self._canvas_to_world(canvas_x, canvas_y)
        new_scale = max(0.01, min(32.0, self.transform.scale * factor))
        self.transform = self.transform.scaled_about(new_scale, world_x, world_y)
        self.image_scale_percent.set(round(new_scale * 100.0, 3))
        self._request_render(preview=True)
        self._update_selection_info()

    def _source_drag_start(self, event: tk.Event) -> None:
        if self.source is not None:
            self.source_drag_origin = (event.x, event.y, self.transform)

    def _source_drag_move(self, event: tk.Event) -> None:
        if self.source_drag_origin is None:
            return
        start_x, start_y, start_transform = self.source_drag_origin
        self.transform = start_transform.translated(
            (event.x - start_x) / self.view_scale,
            (event.y - start_y) / self.view_scale,
        )
        self._request_render(preview=True)
        self._update_selection_info()

    def _source_drag_end(self, _event: tk.Event) -> None:
        self.source_drag_origin = None
        self._request_render(preview=False)

    def _view_drag_start(self, event: tk.Event) -> None:
        self.view_drag_origin = (event.x, event.y, self.view_center_x, self.view_center_y)

    def _view_drag_move(self, event: tk.Event) -> None:
        if self.view_drag_origin is None:
            return
        start_x, start_y, center_x, center_y = self.view_drag_origin
        self.view_center_x = center_x - (event.x - start_x) / self.view_scale
        self.view_center_y = center_y - (event.y - start_y) / self.view_scale
        self._request_render(preview=True)

    def _view_drag_end(self, _event: tk.Event) -> None:
        self.view_drag_origin = None
        self._request_render(preview=False)

    def _selection_start(self, event: tk.Event) -> None:
        cell = self._canvas_to_cell(event.x, event.y)
        self.selection_start = cell
        self.selection = TileSelection(cell[0], cell[1], cell[0], cell[1])
        self._draw_overlays()
        self._update_selection_info()

    def _selection_move(self, event: tk.Event) -> None:
        if self.selection_start is None:
            return
        cell = self._canvas_to_cell(event.x, event.y)
        self.selection = TileSelection(
            self.selection_start[0], self.selection_start[1], cell[0], cell[1]
        )
        self._draw_overlays()
        self._update_selection_info()

    def _selection_end(self, _event: tk.Event) -> None:
        self.selection_start = None
        self._draw_overlays()
        self._update_selection_info()

    def _canvas_to_world(self, x: float, y: float) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            self.view_center_x + (x - width / 2.0) / self.view_scale,
            self.view_center_y + (y - height / 2.0) / self.view_scale,
        )

    def _world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            width / 2.0 + (x - self.view_center_x) * self.view_scale,
            height / 2.0 + (y - self.view_center_y) * self.view_scale,
        )

    def _canvas_to_cell(self, x: float, y: float) -> tuple[int, int]:
        world_x, world_y = self._canvas_to_world(x, y)
        return math.floor(world_x / TILE_SIZE), math.floor(world_y / TILE_SIZE)

    def _request_render(self, *, preview: bool = False) -> None:
        if self.render_job is not None:
            try:
                self.window.after_cancel(self.render_job)
            except tk.TclError:
                pass
        self.render_job = self.window.after(
            20 if preview else 40,
            lambda: self._queue_render(preview),
        )

    def _queue_render(self, preview: bool) -> None:
        self.render_job = None
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.render_generation += 1
        request = (self.render_generation, width, height, preview)
        self.pending_render = request
        if self.render_running:
            return
        self._start_next_render()

    def _start_next_render(self) -> None:
        request = self.pending_render
        self.pending_render = None
        if request is None:
            return
        generation, width, height, preview = request
        source = self.source
        transform = self.transform
        center_x = self.view_center_x
        center_y = self.view_center_y
        view_scale = self.view_scale
        self.render_running = True

        def worker() -> None:
            try:
                ppm, block_size = self._render_ppm(
                    source,
                    transform,
                    center_x,
                    center_y,
                    view_scale,
                    width,
                    height,
                    preview,
                )
                self.render_results.put(("render", (generation, ppm, block_size)))
            except Exception as exc:
                self.render_results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-cropper-render", daemon=True).start()

    @staticmethod
    def _render_ppm(
        source: PixelImage | None,
        transform: SourceTransform,
        center_x: float,
        center_y: float,
        view_scale: float,
        width: int,
        height: int,
        preview: bool,
    ) -> tuple[bytes, int]:
        block_size = 2 if preview and max(width, height) >= 700 else 1
        render_width = max(1, math.ceil(width / block_size))
        render_height = max(1, math.ceil(height / block_size))
        effective_scale = view_scale / block_size
        output = bytearray(render_width * render_height * 3)

        x_sources: list[int] = []
        if source is not None:
            for x in range(render_width):
                canvas_x = x * block_size + block_size * 0.5
                world_x = center_x + (canvas_x - width / 2.0) / view_scale
                x_sources.append(math.floor((world_x - transform.offset_x) / transform.scale))

        for y in range(render_height):
            canvas_y = y * block_size + block_size * 0.5
            world_y = center_y + (canvas_y - height / 2.0) / view_scale
            source_y = -1 if source is None else math.floor((world_y - transform.offset_y) / transform.scale)
            for x in range(render_width):
                index = (y * render_width + x) * 3
                checker = ((x // 12) + (y // 12)) & 1
                background = (45, 49, 55) if checker == 0 else (58, 63, 70)
                if source is None or source_y < 0 or source_y >= source.height:
                    output[index : index + 3] = bytes(background)
                    continue
                source_x = x_sources[x]
                if source_x < 0 or source_x >= source.width:
                    output[index : index + 3] = bytes(background)
                    continue
                source_index = (source_y * source.width + source_x) * source.channels
                red = source.pixels[source_index]
                green = source.pixels[source_index + 1]
                blue = source.pixels[source_index + 2]
                if source.channels == 4:
                    alpha = source.pixels[source_index + 3]
                    inverse = 255 - alpha
                    red = (red * alpha + background[0] * inverse) // 255
                    green = (green * alpha + background[1] * inverse) // 255
                    blue = (blue * alpha + background[2] * inverse) // 255
                output[index] = red
                output[index + 1] = green
                output[index + 2] = blue
        header = f"P6\n{render_width} {render_height}\n255\n".encode("ascii")
        return header + bytes(output), block_size

    def _poll_results(self) -> None:
        try:
            while True:
                kind, payload = self.render_results.get_nowait()
                if kind == "source":
                    path, image = payload  # type: ignore[misc]
                    self._accept_source(path, image)
                elif kind == "render":
                    generation, ppm, block_size = payload  # type: ignore[misc]
                    self.render_running = False
                    if generation == self.render_generation:
                        photo = tk.PhotoImage(master=self.window, data=ppm, format="PPM")
                        if block_size > 1:
                            photo = photo.zoom(block_size, block_size)
                        self.preview_photo = photo
                        self.canvas.delete("preview")
                        self.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("preview",))
                        self.canvas.tag_lower("preview")
                        self._draw_overlays()
                    self._start_next_render()
                elif kind == "exported":
                    paths = payload  # type: ignore[assignment]
                    self.export_running = False
                    self.export_button.configure(state=tk.NORMAL)
                    self.status.set(t("导出完成：{len}张512×512 PNG，已保存到 art/data", len=len(paths)))
                    self._update_next_output_name()
                    messagebox.showinfo(
                        t("裁剪完成"),
                        t("已生成 {len} 张图片。\n\n统一输出目录：\n{value}", len=len(paths), value=Path(paths[0]).parent if paths else ''),
                        parent=self.window,
                    )
                elif kind == "error":
                    self.render_running = False
                    self.export_running = False
                    if self.source is not None:
                        self.export_button.configure(state=tk.NORMAL)
                    messagebox.showerror(t("图片裁剪器"), str(payload), parent=self.window)
                    self.status.set(t("操作失败：{payload}", payload=payload))
                    self._start_next_render()
        except queue.Empty:
            pass
        if self.window.winfo_exists():
            self.window.after(40, self._poll_results)

    def _draw_overlays(self) -> None:
        self.canvas.delete("grid")
        self.canvas.delete("selection")
        self.canvas.delete("guide")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        min_column, max_column, min_row, max_row = visible_grid_range(
            self.view_center_x,
            self.view_center_y,
            self.view_scale,
            width,
            height,
        )
        cell_display = TILE_SIZE * self.view_scale
        for column in range(min_column, max_column + 2):
            x, _ = self._world_to_canvas(column * TILE_SIZE, 0)
            self.canvas.create_line(x, 0, x, height, fill="#7f8a93", width=1, tags=("grid",))
        for row in range(min_row, max_row + 2):
            _, y = self._world_to_canvas(0, row * TILE_SIZE)
            self.canvas.create_line(0, y, width, y, fill="#7f8a93", width=1, tags=("grid",))

        if cell_display >= 58:
            for row in range(min_row, max_row + 1):
                for column in range(min_column, max_column + 1):
                    left, top = self._world_to_canvas(column * TILE_SIZE, row * TILE_SIZE)
                    right, bottom = self._world_to_canvas((column + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)
                    points = (
                        left + (right - left) * 0.25, top,
                        left + (right - left) * 0.75, top,
                        right, top + (bottom - top) * 0.5,
                        left + (right - left) * 0.75, bottom,
                        left + (right - left) * 0.25, bottom,
                        left, top + (bottom - top) * 0.5,
                    )
                    self.canvas.create_polygon(
                        points,
                        fill="",
                        outline="#bbc5cc",
                        width=1,
                        dash=(3, 3),
                        tags=("grid",),
                    )

        left, top = self._world_to_canvas(
            self.selection.min_column * TILE_SIZE,
            self.selection.min_row * TILE_SIZE,
        )
        right, bottom = self._world_to_canvas(
            (self.selection.max_column + 1) * TILE_SIZE,
            (self.selection.max_row + 1) * TILE_SIZE,
        )
        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline="#ffcf4a",
            width=3,
            tags=("selection",),
        )

        guide_length = TILE_SIZE * self.view_scale
        guide_x = 24
        guide_y = 28
        self.canvas.create_line(
            guide_x,
            guide_y,
            guide_x + guide_length,
            guide_y,
            fill="#ffcf4a",
            width=4,
            tags=("guide",),
        )
        self.canvas.create_line(guide_x, guide_y - 7, guide_x, guide_y + 7, fill="#ffcf4a", width=2, tags=("guide",))
        self.canvas.create_line(
            guide_x + guide_length,
            guide_y - 7,
            guide_x + guide_length,
            guide_y + 7,
            fill="#ffcf4a",
            width=2,
            tags=("guide",),
        )
        self.canvas.create_text(
            guide_x,
            guide_y + 13,
            anchor="nw",
            text=t("这一段 = 1格 = 512 px = 3 km"),
            fill="#fff0a8",
            tags=("guide",),
        )
        self.canvas.tag_raise("grid")
        self.canvas.tag_raise("selection")
        self.canvas.tag_raise("guide")

    def _update_selection_info(self) -> None:
        text = t("已选择 {columns}×{rows}，共 {count} 张", columns=self.selection.columns, rows=self.selection.rows, count=self.selection.count)
        if self.source is not None:
            counts = selection_states(self.source, self.transform, self.selection)
            text += (
                t("\n完整覆盖 {counts}，部分超出 {counts2}，完全在源图外 {counts3}", counts=counts['full'], counts2=counts['partial'], counts3=counts['outside'])
            )
            text += t("\n当前一格覆盖源图约 {value:.2f} 像素", value=TILE_SIZE / self.transform.scale)
        self.selection_info.set(text)

    def _update_next_output_name(self) -> None:
        highest = 0
        self.paths.data_root.mkdir(parents=True, exist_ok=True)
        for entry in self.paths.data_root.iterdir():
            if entry.is_file() and entry.suffix.casefold() == ".png" and entry.stem.isdecimal():
                highest = max(highest, int(entry.stem))
        self.next_output_name.set(t("下一张：{value:03d}.png", value=highest + 1))


    def _export(self) -> None:
        if self.source is None:
            messagebox.showwarning(t("图片裁剪器"), t("请先打开源图"), parent=self.window)
            return
        if self.export_running:
            return
        states = selection_states(self.source, self.transform, self.selection)
        if states["outside"]:
            messagebox.showerror(
                t("不能导出"),
                t("选区中有 {states} 格完全位于源图之外。请移动源图或缩小选区。", states=states['outside']),
                parent=self.window,
            )
            return
        if states["partial"]:
            proceed = messagebox.askyesno(
                t("选区部分超出源图"),
                t("有 {states} 格只有部分区域被源图覆盖，超出部分会透明。继续导出吗？", states=states['partial']),
                parent=self.window,
            )
            if not proceed:
                return
        if self.selection.count > 256:
            proceed = messagebox.askyesno(
                t("大量导出"),
                t("将生成 {count} 张512×512 PNG，可能需要一些时间。继续吗？", count=self.selection.count),
                parent=self.window,
            )
            if not proceed:
                return
        source = self.source
        transform = self.transform
        selection = self.selection
        self.export_running = True
        self.export_button.configure(state=tk.DISABLED)
        self.status.set(t("正在导出 {count} 张图片……", count=selection.count))

        def worker() -> None:
            try:
                paths = export_selection(
                    source,
                    transform,
                    selection,
                    self.paths.data_root,
                )
                self.render_results.put(("exported", paths))
            except Exception as exc:
                self.render_results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-cropper-export", daemon=True).start()

    def _close(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass
