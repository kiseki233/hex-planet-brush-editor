from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path


class BrushThumbnailCache:
    def __init__(self, master: tk.Misc, brush_root: str | Path, side: int = 24) -> None:
        self.master = master
        self.brush_root = Path(brush_root)
        self.side = side
        self.width = side * 2
        self.height = max(2, round(math.sqrt(3) * side))
        self._cache: dict[tuple[str, str, int, int], tk.PhotoImage] = {}
        self._missing_cache: dict[int, tk.PhotoImage] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(self, uid: str, relative_path: str, rotation: int) -> tk.PhotoImage:
        return self.get_path(uid, self.brush_root / relative_path, rotation, padding=0)

    def get_path(
        self,
        cache_key: str,
        image_path: str | Path,
        rotation: int,
        padding: int = 0,
    ) -> tk.PhotoImage:
        path = Path(image_path)
        key = (cache_key, str(path), rotation, padding)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            source = tk.PhotoImage(master=self.master, file=str(path))
            image = self._render_hex(source, rotation, padding)
        except tk.TclError:
            image = self.get_missing(rotation)
        self._cache[key] = image
        return image

    def get_missing(self, rotation: int = 0) -> tk.PhotoImage:
        cached = self._missing_cache.get(rotation)
        if cached is not None:
            return cached
        image = tk.PhotoImage(master=self.master, width=self.width, height=self.height)
        for y in range(self.height):
            for x in range(self.width):
                if not self._inside_hex(x + 0.5, y + 0.5):
                    image.transparency_set(x, y, True)
                    continue
                block = ((x // 6) + (y // 6)) % 2
                color = "#d14b5a" if block == 0 else "#2f3136"
                image.put(color, (x, y))
        self._missing_cache[rotation] = image
        return image

    def _render_hex(self, source: tk.PhotoImage, rotation: int, padding: int) -> tk.PhotoImage:
        output = tk.PhotoImage(master=self.master, width=self.width, height=self.height)
        source_width = source.width()
        source_height = source.height()
        if padding < 0 or padding * 2 >= source_width or padding * 2 >= source_height:
            raise tk.TclError("Invalid source padding")
        valid_width = source_width - padding * 2
        valid_height = source_height - padding * 2
        angle = math.radians((rotation % 6) * 60)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        for y in range(self.height):
            for x in range(self.width):
                if not self._inside_hex(x + 0.5, y + 0.5):
                    output.transparency_set(x, y, True)
                    continue

                normalized_x = x / max(1, self.width - 1) - 0.5
                normalized_y = y / max(1, self.height - 1) - 0.5
                source_x_normalized = cosine * normalized_x + sine * normalized_y + 0.5
                source_y_normalized = -sine * normalized_x + cosine * normalized_y + 0.5
                source_x = padding + min(
                    valid_width - 1,
                    max(0, round(source_x_normalized * (valid_width - 1))),
                )
                source_y = padding + min(
                    valid_height - 1,
                    max(0, round(source_y_normalized * (valid_height - 1))),
                )
                try:
                    if source.transparency_get(source_x, source_y):
                        output.transparency_set(x, y, True)
                        continue
                except tk.TclError:
                    pass

                color = source.get(source_x, source_y)
                if isinstance(color, tuple):
                    if len(color) >= 3:
                        output.put(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", (x, y))
                    else:
                        output.put("#000000", (x, y))
                else:
                    output.put(str(color), (x, y))
        return output

    def _inside_hex(self, x: float, y: float) -> bool:
        center_x = self.width / 2
        center_y = self.height / 2
        dx = abs(x - center_x)
        dy = abs(y - center_y)
        if dx > self.side or dy > self.height / 2:
            return False
        return math.sqrt(3) * dx + dy <= math.sqrt(3) * self.side + 0.5
