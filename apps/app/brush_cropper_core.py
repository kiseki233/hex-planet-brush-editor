from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .png_pixels import PixelImage, atomic_write_png

TILE_SIZE = 512


class BrushCropError(ValueError):
    pass


@dataclass(frozen=True)
class SourceTransform:
    """Maps source-image pixels into output-world pixels.

    One output tile is always TILE_SIZE x TILE_SIZE world pixels. ``scale`` is
    the number of output-world pixels occupied by one source pixel.
    """

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise BrushCropError("Image scale must be a positive finite number")
        if not math.isfinite(self.offset_x) or not math.isfinite(self.offset_y):
            raise BrushCropError("Image offset must be finite")

    def source_to_world(self, source_x: float, source_y: float) -> tuple[float, float]:
        return (
            self.offset_x + source_x * self.scale,
            self.offset_y + source_y * self.scale,
        )

    def world_to_source(self, world_x: float, world_y: float) -> tuple[float, float]:
        return (
            (world_x - self.offset_x) / self.scale,
            (world_y - self.offset_y) / self.scale,
        )

    def scaled_about(self, new_scale: float, anchor_world_x: float, anchor_world_y: float) -> "SourceTransform":
        if not math.isfinite(new_scale) or new_scale <= 0.0:
            raise BrushCropError("Image scale must be a positive finite number")
        source_x, source_y = self.world_to_source(anchor_world_x, anchor_world_y)
        return SourceTransform(
            new_scale,
            anchor_world_x - source_x * new_scale,
            anchor_world_y - source_y * new_scale,
        )

    def translated(self, delta_world_x: float, delta_world_y: float) -> "SourceTransform":
        return SourceTransform(
            self.scale,
            self.offset_x + delta_world_x,
            self.offset_y + delta_world_y,
        )


@dataclass(frozen=True)
class TileSelection:
    column_a: int = 0
    row_a: int = 0
    column_b: int = 0
    row_b: int = 0

    @property
    def min_column(self) -> int:
        return min(self.column_a, self.column_b)

    @property
    def max_column(self) -> int:
        return max(self.column_a, self.column_b)

    @property
    def min_row(self) -> int:
        return min(self.row_a, self.row_b)

    @property
    def max_row(self) -> int:
        return max(self.row_a, self.row_b)

    @property
    def columns(self) -> int:
        return self.max_column - self.min_column + 1

    @property
    def rows(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def count(self) -> int:
        return self.columns * self.rows

    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (column, row)
            for row in range(self.min_row, self.max_row + 1)
            for column in range(self.min_column, self.max_column + 1)
        )


def source_world_bounds(source: PixelImage, transform: SourceTransform) -> tuple[float, float, float, float]:
    left = transform.offset_x
    top = transform.offset_y
    right = left + source.width * transform.scale
    bottom = top + source.height * transform.scale
    return left, top, right, bottom


def tile_world_bounds(column: int, row: int) -> tuple[float, float, float, float]:
    left = column * TILE_SIZE
    top = row * TILE_SIZE
    return left, top, left + TILE_SIZE, top + TILE_SIZE


def tile_intersection_state(
    source: PixelImage,
    transform: SourceTransform,
    column: int,
    row: int,
) -> str:
    source_left, source_top, source_right, source_bottom = source_world_bounds(source, transform)
    tile_left, tile_top, tile_right, tile_bottom = tile_world_bounds(column, row)
    intersection_width = min(source_right, tile_right) - max(source_left, tile_left)
    intersection_height = min(source_bottom, tile_bottom) - max(source_top, tile_top)
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return "outside"
    if (
        source_left <= tile_left
        and source_top <= tile_top
        and source_right >= tile_right
        and source_bottom >= tile_bottom
    ):
        return "full"
    return "partial"


def crop_tile_nearest(
    source: PixelImage,
    transform: SourceTransform,
    column: int,
    row: int,
    *,
    tile_size: int = TILE_SIZE,
) -> PixelImage:
    if tile_size < 1:
        raise BrushCropError("Tile size must be positive")
    if source.channels not in (3, 4):
        raise BrushCropError("Only RGB and RGBA source images are supported")

    world_left = column * tile_size
    world_top = row * tile_size
    x_map = [
        math.floor((world_left + x + 0.5 - transform.offset_x) / transform.scale)
        for x in range(tile_size)
    ]
    y_map = [
        math.floor((world_top + y + 0.5 - transform.offset_y) / transform.scale)
        for y in range(tile_size)
    ]

    output = bytearray(tile_size * tile_size * 4)
    source_channels = source.channels
    for target_y, source_y in enumerate(y_map):
        if source_y < 0 or source_y >= source.height:
            continue
        target_row = target_y * tile_size * 4
        source_row = source_y * source.width * source_channels
        for target_x, source_x in enumerate(x_map):
            if source_x < 0 or source_x >= source.width:
                continue
            source_index = source_row + source_x * source_channels
            target_index = target_row + target_x * 4
            output[target_index] = source.pixels[source_index]
            output[target_index + 1] = source.pixels[source_index + 1]
            output[target_index + 2] = source.pixels[source_index + 2]
            output[target_index + 3] = (
                source.pixels[source_index + 3] if source_channels == 4 else 255
            )
    return PixelImage(tile_size, tile_size, 4, bytes(output))


def numbered_output_paths(output_root: str | Path, count: int) -> tuple[Path, ...]:
    """Reserve the next global numeric PNG names in one flat art/data directory.

    Existing files named only with decimal digits and a .png extension participate
    in the sequence. Numbering is zero-padded to at least three digits, so the
    sequence is 001.png, 002.png, ... 999.png, 1000.png, and so on.
    """
    if count < 1:
        raise BrushCropError("Export count must be positive")
    output_directory = Path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    highest = 0
    for entry in output_directory.iterdir():
        if not entry.is_file() or entry.suffix.casefold() != ".png":
            continue
        stem = entry.stem
        if stem.isdecimal():
            highest = max(highest, int(stem))

    paths: list[Path] = []
    number = highest + 1
    while len(paths) < count:
        target = output_directory / f"{number:03d}.png"
        if not target.exists():
            paths.append(target)
        number += 1
    return tuple(paths)


def export_selection(
    source: PixelImage,
    transform: SourceTransform,
    selection: TileSelection,
    output_root: str | Path,
) -> tuple[Path, ...]:
    """Export selected tiles directly into one flat numbered art/data directory."""
    targets = numbered_output_paths(output_root, selection.count)
    written: list[Path] = []
    for (column, row), target in zip(selection.cells(), targets, strict=True):
        image = crop_tile_nearest(source, transform, column, row)
        atomic_write_png(target, image)
        written.append(target)
    return tuple(written)


def visible_grid_range(
    center_world_x: float,
    center_world_y: float,
    view_scale: float,
    width: int,
    height: int,
    *,
    margin_cells: int = 1,
) -> tuple[int, int, int, int]:
    if view_scale <= 0.0:
        raise BrushCropError("View scale must be positive")
    half_world_width = width / (2.0 * view_scale)
    half_world_height = height / (2.0 * view_scale)
    min_column = math.floor((center_world_x - half_world_width) / TILE_SIZE) - margin_cells
    max_column = math.floor((center_world_x + half_world_width) / TILE_SIZE) + margin_cells
    min_row = math.floor((center_world_y - half_world_height) / TILE_SIZE) - margin_cells
    max_row = math.floor((center_world_y + half_world_height) / TILE_SIZE) + margin_cells
    return min_column, max_column, min_row, max_row


def selection_states(
    source: PixelImage,
    transform: SourceTransform,
    selection: TileSelection,
) -> dict[str, int]:
    counts = {"full": 0, "partial": 0, "outside": 0}
    for column, row in selection.cells():
        counts[tile_intersection_state(source, transform, column, row)] += 1
    return counts


def count_existing_brush_pngs(brush_root: str | Path) -> int:
    root = Path(brush_root)
    count = 0
    if not root.exists():
        return 0
    for path in root.rglob("*.png"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if ".catalog" in relative.parts:
            continue
        count += 1
    return count
