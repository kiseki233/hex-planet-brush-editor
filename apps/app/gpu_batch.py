from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .brush_catalog import BrushRecord
from .brush_lod import BrushLodCache
from .chunk_layout import ChunkLayout
from .png_pixels import PixelImage, read_png_pixels
from .sphere_map_store import SphereMapSession, SphereMapStore
from .topology import DualTopology

GPU_BATCH_VERSION = 2
INSTANCE_STRUCT = struct.Struct("<20f")
MESH_VERTEX_STRUCT = struct.Struct("<3f")
EMPTY_LAYER_KEY = "__empty__"
MISSING_LAYER_KEY = "__missing__"


class GpuBatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuTextureLayer:
    layer: int
    key: str
    brush_uid: str | None
    content_hash: str
    source_path: Path | None
    kind: str


@dataclass(frozen=True)
class GpuInstance:
    cell_id: int
    corners: tuple[tuple[float, float, float], ...]
    texture_layer: int
    rotation: int

    def __post_init__(self) -> None:
        if len(self.corners) != 6:
            raise GpuBatchError(f"GpuInstance {self.cell_id} requires exactly six shared corners")

    @property
    def center(self) -> tuple[float, float, float]:
        return _normalize(tuple(sum(corner[axis] for corner in self.corners) for axis in range(3)))

    @property
    def tangent_u(self) -> tuple[float, float, float]:
        center = self.center
        first = self.corners[0]
        projection = tuple(first[axis] - center[axis] * _dot(first, center) for axis in range(3))
        return _normalize(projection)

    @property
    def tangent_v(self) -> tuple[float, float, float]:
        return _normalize(_cross(self.center, self.tangent_u))

    @property
    def radius(self) -> float:
        center = self.center
        return sum(
            math.sqrt(sum((corner[axis] - center[axis]) ** 2 for axis in range(3)))
            for corner in self.corners
        ) / 6.0

    def packed(self) -> bytes:
        values = [component for corner in self.corners for component in corner]
        values.extend((float(self.texture_layer), float(self.rotation)))
        return INSTANCE_STRUCT.pack(*values)


@dataclass(frozen=True)
class GpuRenderBatch:
    version: int
    lod_level: int
    effective_size: int
    padded_size: int
    topology_hash: str
    layout_hash: str
    instances: tuple[GpuInstance, ...]
    texture_layers: tuple[GpuTextureLayer, ...]
    stable_hash: str

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def texture_layer_count(self) -> int:
        return len(self.texture_layers)

    def instance_bytes(self) -> bytes:
        return b"".join(instance.packed() for instance in self.instances)


@dataclass(frozen=True)
class GpuTextureArrayData:
    width: int
    height: int
    layer_count: int
    pixels_rgba: bytes
    layer_keys: tuple[str, ...]

    @property
    def byte_size(self) -> int:
        return len(self.pixels_rgba)


@dataclass(frozen=True)
class GpuTexturePayload:
    key: str
    brush_uid: str
    width: int
    height: int
    pixels_rgba: bytes


class GpuRenderBatchBuilder:
    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self.lod_cache = BrushLodCache(self.brush_root)

    def build(
        self,
        topology: DualTopology,
        layout: ChunkLayout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        cell_ids: Iterable[int],
        lod_level: int,
        *,
        ensure_lod: bool = True,
    ) -> GpuRenderBatch:
        if lod_level < 0 or lod_level > 3:
            raise GpuBatchError("GPU detail rendering supports texture LOD levels 0 through 3")
        if layout.topology_hash != topology.stable_hash:
            raise GpuBatchError("Topology and chunk layout hashes do not match")
        if session.layout.stable_hash != layout.stable_hash:
            raise GpuBatchError("Map session and chunk layout hashes do not match")

        requested = tuple(dict.fromkeys(int(cell_id) for cell_id in cell_ids))
        for cell_id in requested:
            if cell_id < 0 or cell_id >= topology.cell_count:
                raise GpuBatchError(f"CellId is outside topology: {cell_id}")

        empty_layer = GpuTextureLayer(0, EMPTY_LAYER_KEY, None, "", None, "empty")
        missing_layer = GpuTextureLayer(1, MISSING_LAYER_KEY, None, "", None, "missing")
        layers: list[GpuTextureLayer] = [empty_layer, missing_layer]
        layer_by_key: dict[str, int] = {
            EMPTY_LAYER_KEY: empty_layer.layer,
            MISSING_LAYER_KEY: missing_layer.layer,
        }
        instances: list[GpuInstance] = []
        pentagons = set(topology.pentagon_ids)

        chunk_values: dict[int, list[int]] = {}
        for cell_id in requested:
            if cell_id in pentagons:
                continue
            chunk_id, local_index = layout.chunk_for_cell(cell_id)
            values = chunk_values.get(chunk_id)
            if values is None:
                values = session.loaded_chunks.get(chunk_id)
                if values is None:
                    values = store.read_chunk_values(session, chunk_id)
                chunk_values[chunk_id] = values
            value = values[local_index]
            local_id = value & 0x0FFF
            rotation = (value >> 12) & 0x0007
            if rotation > 5:
                raise GpuBatchError(f"CellId {cell_id} has invalid rotation {rotation}")

            if local_id == 0:
                layer = layer_by_key[EMPTY_LAYER_KEY]
            else:
                entry = session.brush_entries.get(local_id)
                record = None if entry is None else records_by_uid.get(entry.brush_uid)
                if record is None or record.state != "active":
                    layer = layer_by_key[MISSING_LAYER_KEY]
                else:
                    key = brush_texture_key(record, lod_level)
                    layer = layer_by_key.get(key, -1)
                    if layer < 0:
                        path = self.lod_cache.existing_path(record, lod_level)
                        if path is None and ensure_lod:
                            path = self.lod_cache.ensure(record, lod_level).path
                        if path is None:
                            raise GpuBatchError(
                                f"Brush LOD is not ready: {record.relative_path}, LOD{lod_level}"
                            )
                        layer = len(layers)
                        layer_by_key[key] = layer
                        layers.append(
                            GpuTextureLayer(
                                layer=layer,
                                key=key,
                                brush_uid=record.uid,
                                content_hash=record.content_hash,
                                source_path=path,
                                kind="brush",
                            )
                        )

            instances.append(
                GpuInstance(
                    cell_id=cell_id,
                    corners=_exact_cell_corners(topology, cell_id),
                    texture_layer=layer,
                    rotation=rotation,
                )
            )

        effective_size = self.lod_cache.effective_size(lod_level)
        padded_size = self.lod_cache.padded_size(lod_level)
        stable_hash = _batch_hash(
            lod_level,
            topology.stable_hash,
            layout.stable_hash,
            instances,
            layers,
        )
        return GpuRenderBatch(
            version=GPU_BATCH_VERSION,
            lod_level=lod_level,
            effective_size=effective_size,
            padded_size=padded_size,
            topology_hash=topology.stable_hash,
            layout_hash=layout.stable_hash,
            instances=tuple(instances),
            texture_layers=tuple(layers),
            stable_hash=stable_hash,
        )


def brush_texture_key(record: BrushRecord, lod_level: int) -> str:
    if lod_level < 0 or lod_level > 3:
        raise GpuBatchError("GPU texture LOD must be between 0 and 3")
    return f"{record.uid}:{record.content_hash}:lod{lod_level}"


def build_brush_texture_payload(
    brush_root: str | Path,
    record: BrushRecord,
    lod_level: int,
) -> GpuTexturePayload:
    if record.state != "active":
        raise GpuBatchError(f"Brush is not active: {record.relative_path}")
    cache = BrushLodCache(brush_root)
    result = cache.ensure(record, lod_level)
    image = read_png_pixels(result.path)
    expected = cache.padded_size(lod_level)
    if image.width != expected or image.height != expected:
        raise GpuBatchError(
            f"Texture layer size mismatch for {result.path}: expected {expected}x{expected}, "
            f"received {image.width}x{image.height}"
        )
    return GpuTexturePayload(
        key=brush_texture_key(record, lod_level),
        brush_uid=record.uid,
        width=image.width,
        height=image.height,
        pixels_rgba=_to_rgba(image),
    )


def build_texture_array(batch: GpuRenderBatch) -> GpuTextureArrayData:
    side = batch.padded_size
    layer_payloads: list[bytes] = []
    keys: list[str] = []
    for layer in batch.texture_layers:
        keys.append(layer.key)
        layer_payloads.append(build_texture_layer_payload(layer, side))
    return GpuTextureArrayData(
        width=side,
        height=side,
        layer_count=len(layer_payloads),
        pixels_rgba=b"".join(layer_payloads),
        layer_keys=tuple(keys),
    )


def build_texture_layer_payload(layer: GpuTextureLayer, side: int) -> bytes:
    """Decode one texture layer without materialising the complete texture array."""
    if layer.kind == "empty":
        payload = _solid_rgba(side, (63, 73, 82, 255))
    elif layer.kind == "missing":
        payload = _missing_rgba(side)
    elif layer.kind == "brush":
        if layer.source_path is None:
            raise GpuBatchError(f"Texture layer {layer.layer} has no source path")
        image = read_png_pixels(layer.source_path)
        if image.width != side or image.height != side:
            raise GpuBatchError(
                f"Texture layer size mismatch for {layer.source_path}: "
                f"expected {side}x{side}, received {image.width}x{image.height}"
            )
        payload = _to_rgba(image)
    else:
        raise GpuBatchError(f"Unknown texture layer kind: {layer.kind}")
    expected = side * side * 4
    if len(payload) != expected:
        raise GpuBatchError(
            f"Texture layer {layer.layer} byte count mismatch: expected {expected}, received {len(payload)}"
        )
    return payload


def unit_hex_mesh_bytes() -> bytes:
    """Return a triangle fan whose first attribute selects an exact shared corner.

    Selector -1 is the cell center.  Selectors 0..5 reference the six per-instance
    spherical dual corners stored in INSTANCE_STRUCT.
    """
    vertices: list[bytes] = []
    corner_uvs: list[tuple[float, float]] = []
    for index in range(6):
        angle = math.radians(60.0 * index)
        corner_uvs.append((math.cos(angle) * 0.5 + 0.5, 0.5 - math.sin(angle) * 0.5))
    for index in range(6):
        next_index = (index + 1) % 6
        vertices.append(MESH_VERTEX_STRUCT.pack(-1.0, 0.5, 0.5))
        vertices.append(MESH_VERTEX_STRUCT.pack(float(index), *corner_uvs[index]))
        vertices.append(MESH_VERTEX_STRUCT.pack(float(next_index), *corner_uvs[next_index]))
    return b"".join(vertices)


def rotate_uv(uv: tuple[float, float], rotation: int) -> tuple[float, float]:
    if rotation < 0 or rotation > 5:
        raise ValueError("rotation must be between 0 and 5")
    x = uv[0] - 0.5
    y = uv[1] - 0.5
    angle = math.radians(rotation * 60.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return 0.5 + x * cosine - y * sine, 0.5 + x * sine + y * cosine


def _exact_cell_corners(topology, cell_id: int) -> tuple[tuple[float, float, float], ...]:
    method = getattr(topology, "cell_corners", None)
    if callable(method):
        corners = tuple(method(cell_id))
    else:
        incident = topology.incident_triangles[cell_id]
        corners = tuple(topology.triangle_centers[triangle_id] for triangle_id in incident)
    if len(corners) != 6:
        raise GpuBatchError(f"CellId {cell_id} does not have six renderable dual corners")
    return corners


def _cell_tangent_basis(
    topology: DualTopology,
    cell_id: int,
    center: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    incident = topology.incident_triangles[cell_id]
    if not incident:
        raise GpuBatchError(f"CellId {cell_id} has no incident triangles")
    first = topology.triangle_centers[incident[0]]
    first_projection = (
        first[0] - center[0] * _dot(first, center),
        first[1] - center[1] * _dot(first, center),
        first[2] - center[2] * _dot(first, center),
    )
    tangent_u = _normalize(first_projection)
    tangent_v = _normalize(_cross(center, tangent_u))
    if len(incident) > 1:
        second = topology.triangle_centers[incident[1]]
        second_projection = (
            second[0] - center[0] * _dot(second, center),
            second[1] - center[1] * _dot(second, center),
            second[2] - center[2] * _dot(second, center),
        )
        if _dot(second_projection, tangent_v) < 0.0:
            tangent_v = (-tangent_v[0], -tangent_v[1], -tangent_v[2])
    return tangent_u, tangent_v


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cell_radius(
    topology: DualTopology,
    cell_id: int,
    center: tuple[float, float, float],
) -> float:
    distances: list[float] = []
    for triangle_id in topology.incident_triangles[cell_id]:
        corner = topology.triangle_centers[triangle_id]
        distances.append(
            math.sqrt(
                (corner[0] - center[0]) ** 2
                + (corner[1] - center[1]) ** 2
                + (corner[2] - center[2]) ** 2
            )
        )
    if not distances:
        raise GpuBatchError(f"CellId {cell_id} has no incident triangles")
    return sum(distances) / len(distances)


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
    if length <= 1e-12:
        raise GpuBatchError("Cannot normalize a zero-length tangent vector")
    return value[0] / length, value[1] / length, value[2] / length


def _solid_rgba(side: int, rgba: tuple[int, int, int, int]) -> bytes:
    return bytes(rgba) * (side * side)


def _missing_rgba(side: int) -> bytes:
    output = bytearray(side * side * 4)
    tile = max(4, side // 8)
    for y in range(side):
        for x in range(side):
            light = ((x // tile) + (y // tile)) % 2 == 0
            color = (208, 69, 173, 255) if light else (55, 29, 48, 255)
            offset = (y * side + x) * 4
            output[offset : offset + 4] = bytes(color)
    return bytes(output)


def _to_rgba(image: PixelImage) -> bytes:
    if image.channels == 4:
        return image.pixels
    if image.channels != 3:
        raise GpuBatchError(f"Unsupported texture channel count: {image.channels}")
    output = bytearray(image.width * image.height * 4)
    source = image.pixels
    target_offset = 0
    for source_offset in range(0, len(source), 3):
        output[target_offset : target_offset + 4] = source[source_offset : source_offset + 3] + b"\xff"
        target_offset += 4
    return bytes(output)


def _batch_hash(
    lod_level: int,
    topology_hash: str,
    layout_hash: str,
    instances: list[GpuInstance],
    layers: list[GpuTextureLayer],
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", GPU_BATCH_VERSION, lod_level))
    digest.update(bytes.fromhex(topology_hash))
    digest.update(bytes.fromhex(layout_hash))
    for layer in layers:
        payload = {
            "layer": layer.layer,
            "key": layer.key,
            "uid": layer.brush_uid,
            "hash": layer.content_hash,
            "kind": layer.kind,
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for instance in instances:
        digest.update(struct.pack("<I", instance.cell_id))
        digest.update(instance.packed())
    return digest.hexdigest()
