from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import struct
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from .brush_catalog import BrushRecord
from .chunk_layout import ChunkLayout
from .map_store import BrushTableEntry
from .png_pixels import (
    PixelImage,
    atomic_write_png,
    read_png_pixels,
    sample_average_rgb,
)
from .sphere_map_store import ChunkIndexRecord, SphereMapSession, SphereMapStore
from .topology import DualTopology

DISTANT_CACHE_VERSION = 1
CHUNK_THUMBNAIL_SIZE = 64
PLANET_OVERVIEW_WIDTH = 512
PLANET_OVERVIEW_HEIGHT = 256
EMPTY_RGB = (46, 56, 66)
MISSING_RGB = (176, 72, 176)


class DistantLodError(RuntimeError):
    pass


@dataclass(frozen=True)
class DistantChunkInfo:
    chunk_id: int
    signature: str
    path: Path
    average_rgb: tuple[int, int, int]
    generated: bool


@dataclass(frozen=True)
class PlanetOverviewInfo:
    signature: str
    path: Path
    width: int
    height: int
    generated: bool


@dataclass(frozen=True)
class DistantBuildResult:
    chunk_id: int
    signature: str | None
    path: Path | None
    average_rgb: tuple[int, int, int] | None
    generated: bool
    error: str | None


@dataclass(frozen=True)
class DistantRequestUpdate:
    ready: tuple[int, ...]
    scheduled: tuple[int, ...]
    pending: int


@dataclass(frozen=True)
class PlanetBuildResult:
    signature: str | None
    path: Path | None
    generated: bool
    error: str | None


@dataclass(frozen=True)
class _ChunkBuildInput:
    session_name: str
    map_dir: Path
    layout: ChunkLayout
    topology: DualTopology
    chunk_id: int
    values: tuple[int, ...] | None
    brush_entries: dict[int, BrushTableEntry]
    brush_records: dict[str, BrushRecord]


@dataclass(frozen=True)
class _PlanetBuildInput:
    session_name: str
    map_dir: Path
    layout: ChunkLayout
    topology: DualTopology
    brush_entries: dict[int, BrushTableEntry]
    brush_records: dict[str, BrushRecord]
    index_records: tuple[ChunkIndexRecord, ...]
    chunks_per_pack: int


class DistantLodCache:
    """Derived chunk thumbnails and an equirectangular planet overview.

    The map's Pack data and brush PNG files remain authoritative. Everything
    written by this class can be deleted and regenerated.
    """

    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self._color_cache: dict[tuple[str, str], tuple[int, int, int]] = {}
        self._color_lock = threading.Lock()

    @staticmethod
    def cache_root(map_dir: Path) -> Path:
        return map_dir / "lod" / "distant_v1"

    @classmethod
    def chunk_path(cls, map_dir: Path, chunk_id: int) -> Path:
        return cls.cache_root(map_dir) / "chunks" / f"chunk_{chunk_id:06d}.png"

    @classmethod
    def chunk_manifest_path(cls, map_dir: Path, chunk_id: int) -> Path:
        return cls.cache_root(map_dir) / "chunks" / f"chunk_{chunk_id:06d}.json"

    @classmethod
    def planet_path(cls, map_dir: Path) -> Path:
        return cls.cache_root(map_dir) / f"planet_{PLANET_OVERVIEW_WIDTH}x{PLANET_OVERVIEW_HEIGHT}.png"

    @classmethod
    def planet_manifest_path(cls, map_dir: Path) -> Path:
        return cls.cache_root(map_dir) / "planet.json"

    def representative_color(self, record: BrushRecord | None) -> tuple[int, int, int]:
        if record is None or record.state != "active":
            return MISSING_RGB
        if (
            record.average_rgb is not None
            and len(record.average_rgb) == 3
            and all(0 <= channel <= 255 for channel in record.average_rgb)
        ):
            return tuple(record.average_rgb)
        key = (record.uid, record.content_hash)
        with self._color_lock:
            cached = self._color_cache.get(key)
        if cached is not None:
            return cached

        source_path = self.brush_root / record.relative_path
        try:
            image = read_png_pixels(source_path)
        except Exception:
            color = MISSING_RGB
        else:
            color = sample_average_rgb(image)
        with self._color_lock:
            self._color_cache[key] = color
        return color

    def chunk_signature(
        self,
        layout: ChunkLayout,
        chunk_id: int,
        values: tuple[int, ...],
        brush_entries: Mapping[int, BrushTableEntry],
        brush_records: Mapping[str, BrushRecord],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<III", DISTANT_CACHE_VERSION, layout.frequency, chunk_id))
        digest.update(bytes.fromhex(layout.topology_hash))
        digest.update(bytes.fromhex(layout.stable_hash))
        digest.update(struct.pack(f"<{len(values)}H", *values))
        used_local_ids = sorted({value & 0x0FFF for value in values if (value & 0x0FFF) != 0})
        for local_id in used_local_ids:
            entry = brush_entries.get(local_id)
            digest.update(struct.pack("<H", local_id))
            if entry is None:
                digest.update(b"missing-entry")
                continue
            digest.update(entry.brush_uid.encode("utf-8"))
            record = brush_records.get(entry.brush_uid)
            if record is None:
                digest.update(b"missing-record")
            else:
                digest.update(record.content_hash.encode("ascii"))
                digest.update(record.state.encode("ascii"))
        return digest.hexdigest()

    def existing_chunk(
        self,
        map_dir: Path,
        chunk_id: int,
        signature: str,
    ) -> DistantChunkInfo | None:
        image_path = self.chunk_path(map_dir, chunk_id)
        manifest_path = self.chunk_manifest_path(map_dir, chunk_id)
        if not image_path.is_file() or not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            average = tuple(int(value) for value in payload["averageRgb"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if (
            payload.get("version") != DISTANT_CACHE_VERSION
            or payload.get("chunkId") != chunk_id
            or payload.get("signature") != signature
            or len(average) != 3
            or any(value < 0 or value > 255 for value in average)
        ):
            return None
        return DistantChunkInfo(chunk_id, signature, image_path, average, False)

    def ensure_chunk(self, build_input: _ChunkBuildInput) -> DistantChunkInfo:
        chunk_id = build_input.chunk_id
        if build_input.values is None:
            raise DistantLodError(f"Chunk {chunk_id} values were not supplied")
        signature = self.chunk_signature(
            build_input.layout,
            chunk_id,
            build_input.values,
            build_input.brush_entries,
            build_input.brush_records,
        )
        existing = self.existing_chunk(build_input.map_dir, chunk_id, signature)
        if existing is not None:
            return existing

        chunk = build_input.layout.chunks[chunk_id]
        if len(chunk.cell_ids) != len(build_input.values):
            raise DistantLodError(
                f"Chunk {chunk_id} cell count mismatch: layout {len(chunk.cell_ids)}, "
                f"values {len(build_input.values)}"
            )
        image, average = self._render_chunk_thumbnail(
            build_input.topology,
            chunk.cell_ids,
            build_input.values,
            build_input.brush_entries,
            build_input.brush_records,
        )
        image_path = self.chunk_path(build_input.map_dir, chunk_id)
        manifest_path = self.chunk_manifest_path(build_input.map_dir, chunk_id)
        atomic_write_png(image_path, image)
        _atomic_write_json(
            manifest_path,
            {
                "version": DISTANT_CACHE_VERSION,
                "map": build_input.session_name,
                "chunkId": chunk_id,
                "signature": signature,
                "size": CHUNK_THUMBNAIL_SIZE,
                "averageRgb": list(average),
                "file": image_path.name,
            },
        )
        return DistantChunkInfo(chunk_id, signature, image_path, average, True)

    def _render_chunk_thumbnail(
        self,
        topology: DualTopology,
        cell_ids: tuple[int, ...],
        values: tuple[int, ...],
        brush_entries: Mapping[int, BrushTableEntry],
        brush_records: Mapping[str, BrushRecord],
    ) -> tuple[PixelImage, tuple[int, int, int]]:
        center = _normalize(
            (
                sum(topology.cell_centers[cell_id][0] for cell_id in cell_ids),
                sum(topology.cell_centers[cell_id][1] for cell_id in cell_ids),
                sum(topology.cell_centers[cell_id][2] for cell_id in cell_ids),
            )
        )
        reference = (0.0, 1.0, 0.0) if abs(center[1]) < 0.9 else (1.0, 0.0, 0.0)
        axis_x = _normalize(_cross(reference, center))
        axis_y = _cross(center, axis_x)
        coordinates = [
            (_dot(topology.cell_centers[cell_id], axis_x), _dot(topology.cell_centers[cell_id], axis_y))
            for cell_id in cell_ids
        ]
        min_x = min(value[0] for value in coordinates)
        max_x = max(value[0] for value in coordinates)
        min_y = min(value[1] for value in coordinates)
        max_y = max(value[1] for value in coordinates)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        margin = 4
        scale = min(
            (CHUNK_THUMBNAIL_SIZE - margin * 2 - 1) / span_x,
            (CHUNK_THUMBNAIL_SIZE - margin * 2 - 1) / span_y,
        )
        used_width = span_x * scale
        used_height = span_y * scale
        origin_x = (CHUNK_THUMBNAIL_SIZE - used_width) / 2.0 - min_x * scale
        origin_y = (CHUNK_THUMBNAIL_SIZE - used_height) / 2.0 + max_y * scale

        pixels = bytearray(CHUNK_THUMBNAIL_SIZE * CHUNK_THUMBNAIL_SIZE * 4)
        color_sum = [0, 0, 0]
        colored_cells = 0
        radius = max(1, min(3, int(round(CHUNK_THUMBNAIL_SIZE / max(8.0, math.sqrt(len(cell_ids)) * 4.0)))))
        for coordinate, value in zip(coordinates, values):
            local_id = value & 0x0FFF
            if local_id == 0:
                color = EMPTY_RGB
            else:
                entry = brush_entries.get(local_id)
                record = None if entry is None else brush_records.get(entry.brush_uid)
                color = self.representative_color(record)
                color_sum[0] += color[0]
                color_sum[1] += color[1]
                color_sum[2] += color[2]
                colored_cells += 1
            x = int(round(origin_x + coordinate[0] * scale))
            y = int(round(origin_y - coordinate[1] * scale))
            _paint_disc_rgba(pixels, CHUNK_THUMBNAIL_SIZE, CHUNK_THUMBNAIL_SIZE, x, y, radius, color)

        average = (
            tuple(round(total / colored_cells) for total in color_sum)
            if colored_cells
            else EMPTY_RGB
        )
        return PixelImage(CHUNK_THUMBNAIL_SIZE, CHUNK_THUMBNAIL_SIZE, 4, bytes(pixels)), average  # type: ignore[arg-type]

    def planet_signature(
        self,
        build_input: _PlanetBuildInput,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            struct.pack(
                "<IIII",
                DISTANT_CACHE_VERSION,
                build_input.layout.frequency,
                PLANET_OVERVIEW_WIDTH,
                PLANET_OVERVIEW_HEIGHT,
            )
        )
        digest.update(bytes.fromhex(build_input.layout.topology_hash))
        digest.update(bytes.fromhex(build_input.layout.stable_hash))
        for record in build_input.index_records:
            digest.update(
                struct.pack(
                    "<IIII",
                    record.chunk_id,
                    record.crc32,
                    record.raw_size,
                    record.cell_count,
                )
            )
        for local_id in sorted(build_input.brush_entries):
            entry = build_input.brush_entries[local_id]
            digest.update(struct.pack("<H", local_id))
            digest.update(entry.brush_uid.encode("utf-8"))
            record = build_input.brush_records.get(entry.brush_uid)
            if record is None:
                digest.update(b"missing-record")
            else:
                digest.update(record.content_hash.encode("ascii"))
                digest.update(record.state.encode("ascii"))
        return digest.hexdigest()

    def existing_planet(self, map_dir: Path, signature: str) -> PlanetOverviewInfo | None:
        image_path = self.planet_path(map_dir)
        manifest_path = self.planet_manifest_path(map_dir)
        if not image_path.is_file() or not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("version") != DISTANT_CACHE_VERSION
            or payload.get("signature") != signature
            or payload.get("width") != PLANET_OVERVIEW_WIDTH
            or payload.get("height") != PLANET_OVERVIEW_HEIGHT
        ):
            return None
        return PlanetOverviewInfo(
            signature,
            image_path,
            PLANET_OVERVIEW_WIDTH,
            PLANET_OVERVIEW_HEIGHT,
            False,
        )

    def ensure_planet(self, build_input: _PlanetBuildInput, store: SphereMapStore) -> PlanetOverviewInfo:
        signature = self.planet_signature(build_input)
        existing = self.existing_planet(build_input.map_dir, signature)
        if existing is not None:
            return existing

        snapshot = SphereMapSession(
            name=build_input.session_name,
            map_dir=build_input.map_dir,
            layout=build_input.layout,
            brush_entries=dict(build_input.brush_entries),
            index_records=list(build_input.index_records),
            chunks_per_pack=build_input.chunks_per_pack,
        )
        pixels = bytearray(PLANET_OVERVIEW_WIDTH * PLANET_OVERVIEW_HEIGHT * 3)
        for index in range(0, len(pixels), 3):
            pixels[index : index + 3] = bytes(EMPTY_RGB)

        radius = max(0, min(2, int(round(PLANET_OVERVIEW_WIDTH / max(256.0, build_input.layout.frequency * 5.5)))))
        for chunk in build_input.layout.chunks:
            values = store.read_chunk_values(snapshot, chunk.chunk_id)
            for cell_id, value in zip(chunk.cell_ids, values):
                local_id = value & 0x0FFF
                if local_id == 0:
                    color = EMPTY_RGB
                else:
                    entry = build_input.brush_entries.get(local_id)
                    record = None if entry is None else build_input.brush_records.get(entry.brush_uid)
                    color = self.representative_color(record)
                x_value, y_value, z_value = build_input.topology.cell_centers[cell_id]
                longitude = math.atan2(z_value, x_value)
                latitude = math.asin(max(-1.0, min(1.0, y_value)))
                x = int(round((longitude + math.pi) / (2.0 * math.pi) * (PLANET_OVERVIEW_WIDTH - 1)))
                y = int(round((math.pi / 2.0 - latitude) / math.pi * (PLANET_OVERVIEW_HEIGHT - 1)))
                _paint_square_rgb_wrap(
                    pixels,
                    PLANET_OVERVIEW_WIDTH,
                    PLANET_OVERVIEW_HEIGHT,
                    x,
                    y,
                    radius,
                    color,
                )

        image_path = self.planet_path(build_input.map_dir)
        atomic_write_png(
            image_path,
            PixelImage(PLANET_OVERVIEW_WIDTH, PLANET_OVERVIEW_HEIGHT, 3, bytes(pixels)),
        )
        _atomic_write_json(
            self.planet_manifest_path(build_input.map_dir),
            {
                "version": DISTANT_CACHE_VERSION,
                "map": build_input.session_name,
                "signature": signature,
                "width": PLANET_OVERVIEW_WIDTH,
                "height": PLANET_OVERVIEW_HEIGHT,
                "file": image_path.name,
                "projection": "equirectangular",
            },
        )
        return PlanetOverviewInfo(
            signature,
            image_path,
            PLANET_OVERVIEW_WIDTH,
            PLANET_OVERVIEW_HEIGHT,
            True,
        )


class AsyncDistantLodBuilder:
    def __init__(
        self,
        cache: DistantLodCache,
        store: SphereMapStore,
        max_workers: int = 2,
        max_inflight: int = 8,
    ) -> None:
        if max_workers < 1 or max_inflight < 1:
            raise ValueError("max_workers and max_inflight must be positive")
        self.cache = cache
        self.store = store
        self.max_inflight = max_inflight
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hexplanet-distant",
        )
        self.session: SphereMapSession | None = None
        self.topology: DualTopology | None = None
        self.brush_records: dict[str, BrushRecord] = {}
        self.required: set[int] = set()
        self.pending: dict[int, concurrent.futures.Future[DistantChunkInfo]] = {}
        self.ready: dict[int, DistantChunkInfo] = {}
        self.failed: dict[int, str] = {}
        self.invalidated: set[int] = set()
        self.closed = False

    def set_context(
        self,
        session: SphereMapSession | None,
        topology: DualTopology | None,
        brush_records: Mapping[str, BrushRecord],
    ) -> None:
        if session is self.session and topology is self.topology and dict(brush_records) == self.brush_records:
            return
        self.session = session
        self.topology = topology
        self.brush_records = dict(brush_records)
        self.required.clear()
        self.ready.clear()
        self.failed.clear()
        self.invalidated.clear()
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()

    def request(
        self,
        session: SphereMapSession,
        topology: DualTopology,
        brush_records: Mapping[str, BrushRecord],
        required_chunk_ids: Iterable[int],
    ) -> DistantRequestUpdate:
        if self.closed:
            raise RuntimeError("AsyncDistantLodBuilder is closed")
        self.set_context(session, topology, brush_records)
        required = set(required_chunk_ids)
        invalid = sorted(chunk_id for chunk_id in required if chunk_id < 0 or chunk_id >= session.layout.chunk_count)
        if invalid:
            raise DistantLodError(f"Distant LOD requested invalid chunk ids: {invalid[:8]}")
        newly_required = required.difference(self.required)
        for chunk_id in newly_required:
            self.failed.pop(chunk_id, None)
        self.required = required
        self.ready = {chunk_id: info for chunk_id, info in self.ready.items() if chunk_id in required}
        scheduled = self._schedule_available()
        return DistantRequestUpdate(
            ready=tuple(sorted(self.ready)),
            scheduled=scheduled,
            pending=len(self.pending),
        )

    def poll(self) -> tuple[DistantBuildResult, ...]:
        results: list[DistantBuildResult] = []
        for chunk_id, future in tuple(self.pending.items()):
            if not future.done():
                continue
            self.pending.pop(chunk_id, None)
            try:
                info = future.result()
            except Exception as exc:
                if chunk_id in self.invalidated:
                    self.invalidated.discard(chunk_id)
                    continue
                message = str(exc)
                self.failed[chunk_id] = message
                results.append(DistantBuildResult(chunk_id, None, None, None, False, message))
                continue
            if chunk_id in self.invalidated:
                self.invalidated.discard(chunk_id)
                continue
            if chunk_id in self.required:
                self.ready[chunk_id] = info
            results.append(
                DistantBuildResult(
                    chunk_id,
                    info.signature,
                    info.path,
                    info.average_rgb,
                    info.generated,
                    None,
                )
            )
        self._schedule_available()
        return tuple(results)

    def invalidate(self, chunk_ids: Iterable[int]) -> None:
        for chunk_id in chunk_ids:
            self.ready.pop(chunk_id, None)
            self.failed.pop(chunk_id, None)
            future = self.pending.get(chunk_id)
            if future is None:
                continue
            if future.cancel():
                self.pending.pop(chunk_id, None)
            else:
                self.invalidated.add(chunk_id)
        self._schedule_available()

    def info(self, chunk_id: int) -> DistantChunkInfo | None:
        return self.ready.get(chunk_id)

    def pending_count(self) -> int:
        return len(self.pending)

    def _build_chunk_worker(
        self,
        build_input: _ChunkBuildInput,
        snapshot: SphereMapSession | None,
    ) -> DistantChunkInfo:
        if build_input.values is None:
            if snapshot is None:
                raise DistantLodError("A Pack snapshot is required for an unloaded chunk")
            values = tuple(self.store.read_chunk_values(snapshot, build_input.chunk_id))
            build_input = replace(build_input, values=values)
        return self.cache.ensure_chunk(build_input)

    def _schedule_available(self) -> tuple[int, ...]:
        session = self.session
        topology = self.topology
        if self.closed or session is None or topology is None:
            return ()
        candidates = sorted(
            self.required.difference(self.ready)
            .difference(self.pending)
            .difference(self.failed)
            .difference(self.invalidated)
        )
        available = self.max_inflight - len(self.pending)
        scheduled: list[int] = []
        for chunk_id in candidates[:available]:
            values = tuple(session.loaded_chunks[chunk_id]) if chunk_id in session.loaded_chunks else None
            snapshot = None
            if values is None:
                snapshot = SphereMapSession(
                    name=session.name,
                    map_dir=session.map_dir,
                    layout=session.layout,
                    brush_entries=dict(session.brush_entries),
                    index_records=list(session.index_records),
                    chunks_per_pack=session.chunks_per_pack,
                )
            build_input = _ChunkBuildInput(
                session.name,
                session.map_dir,
                session.layout,
                topology,
                chunk_id,
                values,
                dict(session.brush_entries),
                dict(self.brush_records),
            )
            self.pending[chunk_id] = self.executor.submit(
                self._build_chunk_worker, build_input, snapshot
            )
            scheduled.append(chunk_id)
        return tuple(scheduled)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()
        self.invalidated.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)


class AsyncPlanetOverviewBuilder:
    def __init__(self, cache: DistantLodCache, store: SphereMapStore) -> None:
        self.cache = cache
        self.store = store
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hexplanet-planet-lod",
        )
        self.pending: concurrent.futures.Future[PlanetOverviewInfo] | None = None
        self.pending_signature: str | None = None
        self.last_info: PlanetOverviewInfo | None = None
        self.last_error: str | None = None
        self.closed = False

    def request(
        self,
        session: SphereMapSession,
        topology: DualTopology,
        brush_records: Mapping[str, BrushRecord],
    ) -> PlanetOverviewInfo | None:
        build_input = _PlanetBuildInput(
            session.name,
            session.map_dir,
            session.layout,
            topology,
            dict(session.brush_entries),
            dict(brush_records),
            tuple(session.index_records),
            session.chunks_per_pack,
        )
        signature = self.cache.planet_signature(build_input)
        existing = self.cache.existing_planet(session.map_dir, signature)
        if existing is not None:
            self.last_info = existing
            self.last_error = None
            return existing
        if self.closed:
            return None
        if self.pending is not None:
            if self.pending_signature == signature:
                return None
            if not self.pending.done():
                return None
            self.poll()
        self.pending_signature = signature
        self.pending = self.executor.submit(self.cache.ensure_planet, build_input, self.store)
        return None

    def poll(self) -> PlanetBuildResult | None:
        future = self.pending
        if future is None or not future.done():
            return None
        self.pending = None
        self.pending_signature = None
        try:
            info = future.result()
        except Exception as exc:
            message = str(exc)
            self.last_error = message
            return PlanetBuildResult(None, None, False, message)
        self.last_info = info
        self.last_error = None
        return PlanetBuildResult(info.signature, info.path, info.generated, None)

    def is_pending(self) -> bool:
        return self.pending is not None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        self.executor.shutdown(wait=False, cancel_futures=True)


def chunk_convex_hulls(
    projected_cells: Iterable[object],
    layout: ChunkLayout,
) -> dict[int, tuple[float, ...]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for cell in projected_cells:
        cell_id = int(getattr(cell, "cell_id"))
        points = tuple(getattr(cell, "points"))
        chunk_id = layout.cell_to_chunk[cell_id]
        bucket = grouped.setdefault(chunk_id, [])
        bucket.extend((float(points[index]), float(points[index + 1])) for index in range(0, len(points), 2))
    result: dict[int, tuple[float, ...]] = {}
    for chunk_id, points in grouped.items():
        hull = _convex_hull(points)
        result[chunk_id] = tuple(coordinate for point in hull for coordinate in point)
    return result


def polygon_center(points: tuple[float, ...]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    xs = points[0::2]
    ys = points[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _convex_hull(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return tuple(unique)

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _paint_disc_rgba(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius_squared = radius * radius
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius_squared:
                continue
            index = (y * width + x) * 4
            pixels[index : index + 4] = bytes((*color, 255))


def _paint_square_rgb_wrap(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for raw_x in range(center_x - radius, center_x + radius + 1):
            x = raw_x % width
            index = (y * width + x) * 3
            pixels[index : index + 3] = bytes(color)


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if length == 0.0:
        raise DistantLodError("Cannot normalize a zero vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
