from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import threading
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .brush_catalog import BrushRecord
from .distant_lod import DistantLodCache, EMPTY_RGB, MISSING_RGB
from .png_pixels import PixelImage, atomic_write_png, read_png_pixels
from .production_layout import ProductionChunkLayout
from .production_visibility import ProductionVisibilityIndex
from .sphere_map_store import SphereMapSession, SphereMapStore

PRODUCTION_SURFACE_VERSION = 1
PRODUCTION_SURFACE_WIDTH = 1024
PRODUCTION_SURFACE_HEIGHT = 512


class ProductionSurfaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductionSurfaceInfo:
    signature: str
    path: Path
    width: int
    height: int
    generated: bool


@dataclass(frozen=True)
class ProductionSurfaceRegion:
    x: int
    y: int
    width: int
    height: int
    pixels_rgb: bytes


class ProductionSurfaceLiveState:
    """Mutable in-memory surface plus an immutable chunk-to-pixel ownership map."""

    def __init__(
        self,
        visibility: ProductionVisibilityIndex,
        image: PixelImage,
    ) -> None:
        if (
            image.width != PRODUCTION_SURFACE_WIDTH
            or image.height != PRODUCTION_SURFACE_HEIGHT
            or image.channels != 3
        ):
            raise ProductionSurfaceError("Live surface must be the 1024x512 RGB cache")
        if visibility.chunk_count <= 0:
            raise ProductionSurfaceError("Live surface requires visible chunks")
        self.width = image.width
        self.height = image.height
        self.lock = threading.RLock()
        self.pixels = bytearray(image.pixels)

        owners = array("i", [-1]) * (self.width * self.height)
        for chunk_id, bound in enumerate(visibility.chunks):
            center_x, center_y, radius_x, radius_y = _chunk_surface_ellipse(
                bound, self.width, self.height
            )
            for pixel_index in _ellipse_pixel_indices(
                self.width,
                self.height,
                center_x,
                center_y,
                radius_x,
                radius_y,
            ):
                # The persisted surface paints chunks in ascending order, so the
                # last chunk touching a pixel owns its final visible color.
                owners[pixel_index] = chunk_id

        owned_pixels = [array("I") for _ in range(visibility.chunk_count)]
        for pixel_index, chunk_id in enumerate(owners):
            if chunk_id >= 0:
                owned_pixels[chunk_id].append(pixel_index)
        self.owned_pixels = tuple(owned_pixels)

    def replace_image(self, image: PixelImage) -> None:
        if (
            image.width != self.width
            or image.height != self.height
            or image.channels != 3
        ):
            raise ProductionSurfaceError("Replacement surface dimensions do not match")
        with self.lock:
            self.pixels[:] = image.pixels

    def update_chunks(
        self,
        chunk_values: Mapping[int, tuple[int, ...]],
        local_colors: Mapping[int, tuple[int, int, int]],
    ) -> tuple[ProductionSurfaceRegion | None, PixelImage]:
        with self.lock:
            min_x = self.width
            min_y = self.height
            max_x = max_y = -1
            for raw_chunk_id, values in chunk_values.items():
                chunk_id = int(raw_chunk_id)
                if chunk_id < 0 or chunk_id >= len(self.owned_pixels):
                    continue
                color = _average_chunk_color(values, local_colors)
                payload = bytes(color)
                for pixel_index in self.owned_pixels[chunk_id]:
                    offset = pixel_index * 3
                    self.pixels[offset : offset + 3] = payload
                    y, x = divmod(pixel_index, self.width)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)

            image = PixelImage(self.width, self.height, 3, bytes(self.pixels))
            if max_x < min_x or max_y < min_y:
                return None, image

            region_width = max_x - min_x + 1
            region_height = max_y - min_y + 1
            region_pixels = bytearray(region_width * region_height * 3)
            target = 0
            for y in range(min_y, max_y + 1):
                source = (y * self.width + min_x) * 3
                size = region_width * 3
                region_pixels[target : target + size] = self.pixels[
                    source : source + size
                ]
                target += size
            return (
                ProductionSurfaceRegion(
                    min_x,
                    min_y,
                    region_width,
                    region_height,
                    bytes(region_pixels),
                ),
                image,
            )


class ProductionSurfaceCache:
    """Regenerable equirectangular surface used by the software globe view."""

    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self.color_cache = DistantLodCache(self.brush_root)

    def local_colors(
        self,
        session: SphereMapSession,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> dict[int, tuple[int, int, int]]:
        colors: dict[int, tuple[int, int, int]] = {0: EMPTY_RGB}
        for local_id, entry in session.brush_entries.items():
            record = records_by_uid.get(entry.brush_uid)
            colors[local_id] = (
                self.color_cache.representative_color(record)
                if record is not None
                else MISSING_RGB
            )
        return colors

    @staticmethod
    def root(map_dir: Path) -> Path:
        return Path(map_dir) / "lod" / "software_surface_v1"

    @classmethod
    def image_path(cls, map_dir: Path) -> Path:
        return cls.root(map_dir) / f"planet_{PRODUCTION_SURFACE_WIDTH}x{PRODUCTION_SURFACE_HEIGHT}.png"

    @classmethod
    def manifest_path(cls, map_dir: Path) -> Path:
        return cls.root(map_dir) / "surface.json"

    def signature(
        self,
        layout: ProductionChunkLayout,
        visibility: ProductionVisibilityIndex,
        session: SphereMapSession,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"hexplanet-production-software-surface-v1")
        digest.update(struct.pack("<III", PRODUCTION_SURFACE_VERSION, PRODUCTION_SURFACE_WIDTH, PRODUCTION_SURFACE_HEIGHT))
        digest.update(bytes.fromhex(layout.stable_hash))
        digest.update(bytes.fromhex(visibility.stable_hash))
        for record in session.index_records:
            digest.update(
                struct.pack(
                    "<IIIII",
                    record.chunk_id,
                    record.crc32,
                    record.raw_size,
                    record.cell_count,
                    record.compression_type,
                )
            )
        for local_id in sorted(session.brush_entries):
            entry = session.brush_entries[local_id]
            digest.update(struct.pack("<H", local_id))
            digest.update(entry.brush_uid.encode("utf-8"))
            record = records_by_uid.get(entry.brush_uid)
            if record is None:
                digest.update(b"missing-record")
            else:
                digest.update(record.content_hash.encode("ascii"))
                digest.update(record.state.encode("ascii"))
        return digest.hexdigest()

    def existing(self, map_dir: Path, signature: str) -> ProductionSurfaceInfo | None:
        image_path = self.image_path(map_dir)
        manifest_path = self.manifest_path(map_dir)
        if not image_path.is_file() or not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("version") != PRODUCTION_SURFACE_VERSION
            or payload.get("signature") != signature
            or payload.get("width") != PRODUCTION_SURFACE_WIDTH
            or payload.get("height") != PRODUCTION_SURFACE_HEIGHT
            or payload.get("layoutHash") is None
        ):
            return None
        return ProductionSurfaceInfo(
            signature=signature,
            path=image_path,
            width=PRODUCTION_SURFACE_WIDTH,
            height=PRODUCTION_SURFACE_HEIGHT,
            generated=False,
        )

    def ensure(
        self,
        layout: ProductionChunkLayout,
        visibility: ProductionVisibilityIndex,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> tuple[ProductionSurfaceInfo, PixelImage]:
        signature = self.signature(layout, visibility, session, records_by_uid)
        existing = self.existing(session.map_dir, signature)
        if existing is not None:
            return existing, read_png_pixels(existing.path)

        if len(session.index_records) != layout.chunk_count:
            raise ProductionSurfaceError("Map index count does not match production layout")
        if visibility.chunk_count != layout.chunk_count:
            raise ProductionSurfaceError("Visibility chunk count does not match production layout")

        snapshot = SphereMapSession(
            name=session.name,
            map_dir=session.map_dir,
            layout=layout,
            brush_entries=dict(session.brush_entries),
            index_records=list(session.index_records),
            chunks_per_pack=session.chunks_per_pack,
        )
        local_colors = self.local_colors(snapshot, records_by_uid)

        pixels = bytearray(bytes(EMPTY_RGB) * (PRODUCTION_SURFACE_WIDTH * PRODUCTION_SURFACE_HEIGHT))
        raw_average_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for chunk_id, bound in enumerate(visibility.chunks):
            index_record = snapshot.index_records[chunk_id]
            raw_key = (index_record.crc32, index_record.raw_size, index_record.cell_count)
            color = raw_average_cache.get(raw_key)
            if color is None:
                values = store.read_chunk_values(snapshot, chunk_id)
                color = _average_chunk_color(values, local_colors)
                raw_average_cache[raw_key] = color
            center_x, center_y, radius_x, radius_y = _chunk_surface_ellipse(
                bound,
                PRODUCTION_SURFACE_WIDTH,
                PRODUCTION_SURFACE_HEIGHT,
            )
            _paint_ellipse_rgb_wrap(
                pixels,
                PRODUCTION_SURFACE_WIDTH,
                PRODUCTION_SURFACE_HEIGHT,
                center_x,
                center_y,
                radius_x,
                radius_y,
                color,
            )

        image = PixelImage(PRODUCTION_SURFACE_WIDTH, PRODUCTION_SURFACE_HEIGHT, 3, bytes(pixels))
        path = self.image_path(session.map_dir)
        atomic_write_png(path, image)
        _atomic_write_json(
            self.manifest_path(session.map_dir),
            {
                "version": PRODUCTION_SURFACE_VERSION,
                "format": "HEX_PLANET_PRODUCTION_SOFTWARE_SURFACE",
                "signature": signature,
                "width": PRODUCTION_SURFACE_WIDTH,
                "height": PRODUCTION_SURFACE_HEIGHT,
                "layoutHash": layout.stable_hash,
                "visibilityHash": visibility.stable_hash,
                "projection": "equirectangular",
                "file": path.name,
            },
        )
        return (
            ProductionSurfaceInfo(
                signature=signature,
                path=path,
                width=PRODUCTION_SURFACE_WIDTH,
                height=PRODUCTION_SURFACE_HEIGHT,
                generated=True,
            ),
            image,
        )


def _average_chunk_color(
    values: list[int] | tuple[int, ...],
    local_colors: Mapping[int, tuple[int, int, int]],
) -> tuple[int, int, int]:
    if not values:
        return EMPTY_RGB
    counts = Counter(int(value) & 0x0FFF for value in values)
    red = green = blue = 0
    for local_id, count in counts.items():
        color = local_colors.get(local_id, MISSING_RGB)
        red += color[0] * count
        green += color[1] * count
        blue += color[2] * count
    total = len(values)
    return round(red / total), round(green / total), round(blue / total)


def _chunk_surface_ellipse(bound, width: int, height: int) -> tuple[int, int, int, int]:
    longitude = math.atan2(bound.center[2], bound.center[0])
    latitude = math.asin(max(-1.0, min(1.0, bound.center[1])))
    center_x = int(round((longitude + math.pi) / (2.0 * math.pi) * (width - 1)))
    center_y = int(round((math.pi / 2.0 - latitude) / math.pi * (height - 1)))
    latitude_scale = max(0.18, math.cos(latitude))
    radius_x = max(
        1,
        min(
            24,
            int(
                math.ceil(
                    bound.angular_radius
                    / (2.0 * math.pi)
                    * width
                    * 1.65
                    / latitude_scale
                )
            ),
        ),
    )
    radius_y = max(
        1,
        min(
            12,
            int(math.ceil(bound.angular_radius / math.pi * height * 1.65)),
        ),
    )
    return center_x, center_y, radius_x, radius_y


def _ellipse_pixel_indices(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
):
    radius_x = max(1, int(radius_x))
    radius_y = max(1, int(radius_y))
    inverse_x = 1.0 / (radius_x * radius_x)
    inverse_y = 1.0 / (radius_y * radius_y)
    for y in range(max(0, center_y - radius_y), min(height, center_y + radius_y + 1)):
        dy = y - center_y
        remaining = 1.0 - dy * dy * inverse_y
        if remaining < 0.0:
            continue
        span = max(0, int(math.ceil(radius_x * math.sqrt(remaining))))
        for raw_x in range(center_x - span, center_x + span + 1):
            yield y * width + raw_x % width


def _paint_ellipse_rgb_wrap(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    color: tuple[int, int, int],
) -> None:
    payload = bytes(color)
    for pixel_index in _ellipse_pixel_indices(
        width, height, center_x, center_y, radius_x, radius_y
    ):
        offset = pixel_index * 3
        pixels[offset : offset + 3] = payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
