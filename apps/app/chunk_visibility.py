from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunk_layout import ChunkLayout
from .topology import DualTopology

VISIBILITY_FORMAT = "HEX_PLANET_CHUNK_VISIBILITY"
VISIBILITY_VERSION = 1
VISIBILITY_MAGIC = b"HPVIS1\0\0"
VISIBILITY_HEADER = struct.Struct("<8s10I32s32s")
CHUNK_BOUNDS_RECORD = struct.Struct("<I4dII")
VISIBILITY_NODE_RECORD = struct.Struct("<I4diiII")
DEFAULT_LEAF_SIZE = 8


class ChunkVisibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkBounds:
    chunk_id: int
    center: tuple[float, float, float]
    angular_radius: float
    boundary_triangle_ids: tuple[int, ...]


@dataclass(frozen=True)
class VisibilityNode:
    node_id: int
    center: tuple[float, float, float]
    angular_radius: float
    left: int
    right: int
    chunk_ids: tuple[int, ...]

    @property
    def is_leaf(self) -> bool:
        return self.left < 0 and self.right < 0


@dataclass(frozen=True)
class VisibilityQueryResult:
    chunk_ids: tuple[int, ...]
    visited_nodes: int
    rejected_nodes: int
    tested_chunks: int
    candidate_cells: int


@dataclass(frozen=True)
class ChunkVisibilityValidation:
    valid: bool
    chunk_count: int
    node_count: int
    leaf_count: int
    boundary_point_count: int
    stable_hash: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ChunkVisibilityIndex:
    frequency: int
    topology_hash: str
    layout_hash: str
    leaf_size: int
    chunks: tuple[ChunkBounds, ...]
    nodes: tuple[VisibilityNode, ...]
    root_id: int
    stable_hash: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def candidate_cell_ids(
        self, layout: ChunkLayout, chunk_ids: Iterable[int]
    ) -> tuple[int, ...]:
        values: list[int] = []
        for chunk_id in chunk_ids:
            if chunk_id < 0 or chunk_id >= layout.chunk_count:
                raise ChunkVisibilityError(f"Invalid ChunkId in visibility result: {chunk_id}")
            values.extend(layout.chunks[chunk_id].cell_ids)
        return tuple(values)

    def query(
        self,
        layout: ChunkLayout,
        yaw: float,
        pitch: float,
        zoom: float,
        width: int,
        height: int,
        margin: float = 12.0,
    ) -> VisibilityQueryResult:
        if layout.stable_hash != self.layout_hash:
            raise ChunkVisibilityError("Chunk layout hash mismatch")
        width = max(1, int(width))
        height = max(1, int(height))
        base_radius = max(20.0, min(width, height) * 0.43)
        sphere_radius = base_radius * zoom
        center_x = width / 2.0
        center_y = height / 2.0

        candidates: list[int] = []
        visited_nodes = 0
        rejected_nodes = 0
        tested_chunks = 0
        stack = [self.root_id]
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            visited_nodes += 1
            if not _cap_intersects_viewport(
                node.center,
                node.angular_radius,
                yaw,
                pitch,
                sphere_radius,
                center_x,
                center_y,
                width,
                height,
                margin,
            ):
                rejected_nodes += 1
                continue
            if node.is_leaf:
                for chunk_id in node.chunk_ids:
                    tested_chunks += 1
                    bound = self.chunks[chunk_id]
                    if _cap_intersects_viewport(
                        bound.center,
                        bound.angular_radius,
                        yaw,
                        pitch,
                        sphere_radius,
                        center_x,
                        center_y,
                        width,
                        height,
                        margin,
                    ):
                        candidates.append(chunk_id)
            else:
                if node.right >= 0:
                    stack.append(node.right)
                if node.left >= 0:
                    stack.append(node.left)

        chunk_ids = tuple(sorted(candidates))
        candidate_cells = sum(len(layout.chunks[chunk_id].cell_ids) for chunk_id in chunk_ids)
        return VisibilityQueryResult(
            chunk_ids=chunk_ids,
            visited_nodes=visited_nodes,
            rejected_nodes=rejected_nodes,
            tested_chunks=tested_chunks,
            candidate_cells=candidate_cells,
        )

    def validate(
        self,
        topology: DualTopology | None = None,
        layout: ChunkLayout | None = None,
    ) -> ChunkVisibilityValidation:
        issues: list[str] = []
        if self.leaf_size < 1:
            issues.append("leaf size must be positive")
        if not self.nodes:
            issues.append("visibility hierarchy has no nodes")
        if self.root_id < 0 or self.root_id >= len(self.nodes):
            issues.append("root node id is invalid")
        if any(chunk.chunk_id != index for index, chunk in enumerate(self.chunks)):
            issues.append("chunk bounds ids are not contiguous")
        if any(node.node_id != index for index, node in enumerate(self.nodes)):
            issues.append("visibility node ids are not contiguous")

        if topology is not None:
            if topology.frequency != self.frequency:
                issues.append("topology frequency mismatch")
            if topology.stable_hash != self.topology_hash:
                issues.append("topology hash mismatch")
        if layout is not None:
            if layout.frequency != self.frequency:
                issues.append("layout frequency mismatch")
            if layout.stable_hash != self.layout_hash:
                issues.append("layout hash mismatch")
            if layout.chunk_count != self.chunk_count:
                issues.append("layout chunk count mismatch")

        for chunk in self.chunks:
            if not _unit_vector(chunk.center):
                issues.append(f"chunk {chunk.chunk_id} has invalid center")
            if chunk.angular_radius < 0.0 or chunk.angular_radius > math.pi + 1e-9:
                issues.append(f"chunk {chunk.chunk_id} has invalid angular radius")
            if topology is not None:
                for triangle_id in chunk.boundary_triangle_ids:
                    if triangle_id < 0 or triangle_id >= topology.triangle_count:
                        issues.append(
                            f"chunk {chunk.chunk_id} has invalid boundary triangle {triangle_id}"
                        )
                        break

        seen_nodes: set[int] = set()
        seen_chunks: list[int] = []
        visiting: set[int] = set()

        def walk(node_id: int) -> None:
            if node_id in visiting:
                issues.append("visibility hierarchy contains a cycle")
                return
            if node_id in seen_nodes:
                issues.append(f"visibility node {node_id} has more than one parent")
                return
            if node_id < 0 or node_id >= len(self.nodes):
                issues.append(f"visibility hierarchy references invalid node {node_id}")
                return
            visiting.add(node_id)
            seen_nodes.add(node_id)
            node = self.nodes[node_id]
            if not _unit_vector(node.center):
                issues.append(f"visibility node {node_id} has invalid center")
            if node.angular_radius < 0.0 or node.angular_radius > math.pi + 1e-9:
                issues.append(f"visibility node {node_id} has invalid angular radius")
            if node.is_leaf:
                if not node.chunk_ids:
                    issues.append(f"visibility leaf {node_id} is empty")
                if len(node.chunk_ids) > self.leaf_size:
                    issues.append(f"visibility leaf {node_id} exceeds leaf size")
                if node.left != -1 or node.right != -1:
                    issues.append(f"visibility leaf {node_id} has child ids")
                for chunk_id in node.chunk_ids:
                    if chunk_id < 0 or chunk_id >= self.chunk_count:
                        issues.append(f"visibility leaf {node_id} references invalid chunk {chunk_id}")
                        continue
                    seen_chunks.append(chunk_id)
                    bound = self.chunks[chunk_id]
                    if not _cap_contains(
                        node.center,
                        node.angular_radius,
                        bound.center,
                        bound.angular_radius,
                    ):
                        issues.append(f"visibility leaf {node_id} does not contain chunk {chunk_id}")
            else:
                if node.chunk_ids:
                    issues.append(f"visibility internal node {node_id} stores chunk ids")
                if node.left < 0 or node.right < 0:
                    issues.append(f"visibility internal node {node_id} is missing a child")
                for child_id in (node.left, node.right):
                    walk(child_id)
            visiting.remove(node_id)

        if 0 <= self.root_id < len(self.nodes):
            walk(self.root_id)
        if len(seen_nodes) != len(self.nodes):
            issues.append(f"visibility hierarchy leaves {len(self.nodes) - len(seen_nodes)} nodes unused")
        if sorted(seen_chunks) != list(range(self.chunk_count)):
            issues.append("visibility hierarchy does not cover every chunk exactly once")

        expected_hash = _stable_visibility_hash(
            self.frequency,
            self.topology_hash,
            self.layout_hash,
            self.leaf_size,
            self.chunks,
            self.nodes,
            self.root_id,
        )
        if expected_hash != self.stable_hash:
            issues.append("visibility stable hash mismatch")

        return ChunkVisibilityValidation(
            valid=not issues,
            chunk_count=self.chunk_count,
            node_count=self.node_count,
            leaf_count=sum(node.is_leaf for node in self.nodes),
            boundary_point_count=sum(len(chunk.boundary_triangle_ids) for chunk in self.chunks),
            stable_hash=self.stable_hash,
            issues=tuple(issues),
        )


@dataclass(frozen=True)
class ChunkVisibilityCacheInfo:
    directory: Path
    manifest_path: Path
    index_path: Path
    crc32: str
    sha256: str


def build_chunk_visibility_index(
    topology: DualTopology,
    layout: ChunkLayout,
    leaf_size: int = DEFAULT_LEAF_SIZE,
) -> ChunkVisibilityIndex:
    if topology.stable_hash != layout.topology_hash:
        raise ChunkVisibilityError("Topology and layout identity do not match")
    if leaf_size < 1 or leaf_size > 1024:
        raise ChunkVisibilityError("Visibility leaf size must be between 1 and 1024")

    chunks: list[ChunkBounds] = []
    for definition in layout.chunks:
        member_ids = definition.cell_ids
        member_set = set(member_ids)
        point_ids: set[int] = set()
        boundary_ids: set[int] = set()
        for cell_id in member_ids:
            for triangle_id in topology.incident_triangles[cell_id]:
                point_ids.add(triangle_id)
                triangle = topology.triangles[triangle_id]
                if any(vertex not in member_set for vertex in triangle):
                    boundary_ids.add(triangle_id)
        if len(boundary_ids) < 3:
            boundary_ids = point_ids
        cap_points = [topology.cell_centers[cell_id] for cell_id in member_ids]
        cap_points.extend(topology.triangle_centers[triangle_id] for triangle_id in point_ids)
        center, angular_radius = _cap_from_points(cap_points)
        chunks.append(
            ChunkBounds(
                chunk_id=definition.chunk_id,
                center=center,
                angular_radius=angular_radius,
                boundary_triangle_ids=tuple(sorted(boundary_ids)),
            )
        )

    nodes: list[VisibilityNode | None] = []

    def build_node(chunk_ids: tuple[int, ...]) -> int:
        node_id = len(nodes)
        nodes.append(None)
        center, angular_radius = _cap_from_bounds(chunks, chunk_ids)
        if len(chunk_ids) <= leaf_size:
            nodes[node_id] = VisibilityNode(
                node_id=node_id,
                center=center,
                angular_radius=angular_radius,
                left=-1,
                right=-1,
                chunk_ids=tuple(sorted(chunk_ids)),
            )
            return node_id

        axis = _widest_axis(chunks, chunk_ids)
        ordered = tuple(sorted(chunk_ids, key=lambda item: (chunks[item].center[axis], item)))
        midpoint = len(ordered) // 2
        left_id = build_node(ordered[:midpoint])
        right_id = build_node(ordered[midpoint:])
        nodes[node_id] = VisibilityNode(
            node_id=node_id,
            center=center,
            angular_radius=angular_radius,
            left=left_id,
            right=right_id,
            chunk_ids=(),
        )
        return node_id

    root_id = build_node(tuple(range(layout.chunk_count)))
    final_nodes = tuple(node for node in nodes if node is not None)
    final_chunks = tuple(chunks)
    stable_hash = _stable_visibility_hash(
        topology.frequency,
        topology.stable_hash,
        layout.stable_hash,
        leaf_size,
        final_chunks,
        final_nodes,
        root_id,
    )
    index = ChunkVisibilityIndex(
        frequency=topology.frequency,
        topology_hash=topology.stable_hash,
        layout_hash=layout.stable_hash,
        leaf_size=leaf_size,
        chunks=final_chunks,
        nodes=final_nodes,
        root_id=root_id,
        stable_hash=stable_hash,
    )
    validation = index.validate(topology, layout)
    if not validation.valid:
        raise ChunkVisibilityError(
            "Chunk visibility validation failed: " + "; ".join(validation.issues[:8])
        )
    return index


def write_chunk_visibility_cache(
    index: ChunkVisibilityIndex,
    topology_root: str | Path,
) -> ChunkVisibilityCacheInfo:
    directory = Path(topology_root) / f"ico_dual_f{index.frequency}_v1"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "visibility.idx"
    manifest_path = directory / "visibility.json"

    boundary_ids: list[int] = []
    chunk_records = bytearray()
    for chunk in index.chunks:
        offset = len(boundary_ids)
        boundary_ids.extend(chunk.boundary_triangle_ids)
        chunk_records.extend(
            CHUNK_BOUNDS_RECORD.pack(
                chunk.chunk_id,
                chunk.center[0],
                chunk.center[1],
                chunk.center[2],
                chunk.angular_radius,
                offset,
                len(chunk.boundary_triangle_ids),
            )
        )

    leaf_members: list[int] = []
    node_records = bytearray()
    for node in index.nodes:
        offset = len(leaf_members)
        leaf_members.extend(node.chunk_ids)
        node_records.extend(
            VISIBILITY_NODE_RECORD.pack(
                node.node_id,
                node.center[0],
                node.center[1],
                node.center[2],
                node.angular_radius,
                node.left,
                node.right,
                offset,
                len(node.chunk_ids),
            )
        )

    header = VISIBILITY_HEADER.pack(
        VISIBILITY_MAGIC,
        VISIBILITY_VERSION,
        index.frequency,
        index.chunk_count,
        index.node_count,
        index.root_id,
        index.leaf_size,
        VISIBILITY_NODE_RECORD.size,
        CHUNK_BOUNDS_RECORD.size,
        len(boundary_ids),
        len(leaf_members),
        bytes.fromhex(index.topology_hash),
        bytes.fromhex(index.layout_hash),
    )
    payload = bytearray(header)
    payload.extend(chunk_records)
    payload.extend(node_records)
    if boundary_ids:
        payload.extend(struct.pack(f"<{len(boundary_ids)}I", *boundary_ids))
    if leaf_members:
        payload.extend(struct.pack(f"<{len(leaf_members)}I", *leaf_members))
    raw = bytes(payload)
    crc32 = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    sha256 = hashlib.sha256(raw).hexdigest()
    _atomic_write_bytes(index_path, raw)
    _atomic_write_json(
        manifest_path,
        {
            "format": VISIBILITY_FORMAT,
            "formatVersion": VISIBILITY_VERSION,
            "frequency": index.frequency,
            "topologyHash": index.topology_hash,
            "layoutHash": index.layout_hash,
            "visibilityHash": index.stable_hash,
            "leafSize": index.leaf_size,
            "counts": {
                "chunks": index.chunk_count,
                "nodes": index.node_count,
                "leaves": sum(node.is_leaf for node in index.nodes),
                "boundaryTriangleReferences": len(boundary_ids),
            },
            "files": {
                "index": {
                    "path": index_path.name,
                    "bytes": len(raw),
                    "crc32": crc32,
                    "sha256": sha256,
                }
            },
            "bounds": {
                "type": "spherical_cap",
                "hierarchy": "deterministic_binary_median_split",
                "leafMembers": "chunk_ids",
                "distantGeometry": "chunk_boundary_triangle_centers",
            },
        },
    )
    return ChunkVisibilityCacheInfo(directory, manifest_path, index_path, crc32, sha256)


def load_chunk_visibility_cache(directory: str | Path) -> ChunkVisibilityIndex:
    cache_dir = Path(directory)
    try:
        manifest = json.loads((cache_dir / "visibility.json").read_text(encoding="utf-8"))
        payload = (cache_dir / str(manifest["files"]["index"]["path"])).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ChunkVisibilityError(f"Cannot read chunk visibility cache: {exc}") from exc

    if manifest.get("format") != VISIBILITY_FORMAT:
        raise ChunkVisibilityError("Unsupported chunk visibility manifest format")
    if int(manifest.get("formatVersion", -1)) != VISIBILITY_VERSION:
        raise ChunkVisibilityError("Unsupported chunk visibility manifest version")

    actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_crc != str(manifest["files"]["index"].get("crc32", "")).lower():
        raise ChunkVisibilityError("Chunk visibility CRC mismatch")
    if actual_sha != str(manifest["files"]["index"].get("sha256", "")).lower():
        raise ChunkVisibilityError("Chunk visibility SHA-256 mismatch")
    if len(payload) < VISIBILITY_HEADER.size:
        raise ChunkVisibilityError("Chunk visibility header is truncated")

    unpacked = VISIBILITY_HEADER.unpack_from(payload)
    (
        magic,
        version,
        frequency,
        chunk_count,
        node_count,
        root_id,
        leaf_size,
        node_record_size,
        chunk_record_size,
        boundary_count,
        leaf_member_count,
        topology_hash_raw,
        layout_hash_raw,
    ) = unpacked
    if magic != VISIBILITY_MAGIC or version != VISIBILITY_VERSION:
        raise ChunkVisibilityError("Unsupported chunk visibility format")
    if node_record_size != VISIBILITY_NODE_RECORD.size:
        raise ChunkVisibilityError("Unsupported visibility node record size")
    if chunk_record_size != CHUNK_BOUNDS_RECORD.size:
        raise ChunkVisibilityError("Unsupported chunk bounds record size")

    expected_size = (
        VISIBILITY_HEADER.size
        + chunk_count * CHUNK_BOUNDS_RECORD.size
        + node_count * VISIBILITY_NODE_RECORD.size
        + boundary_count * 4
        + leaf_member_count * 4
    )
    if len(payload) != expected_size:
        raise ChunkVisibilityError(
            f"Chunk visibility size mismatch: expected {expected_size}, received {len(payload)}"
        )

    offset = VISIBILITY_HEADER.size
    raw_chunks: list[tuple[int, tuple[float, float, float], float, int, int]] = []
    for _ in range(chunk_count):
        chunk_id, x, y, z, radius, member_offset, member_count = CHUNK_BOUNDS_RECORD.unpack_from(
            payload, offset
        )
        offset += CHUNK_BOUNDS_RECORD.size
        raw_chunks.append((chunk_id, (x, y, z), radius, member_offset, member_count))

    raw_nodes: list[tuple[int, tuple[float, float, float], float, int, int, int, int]] = []
    for _ in range(node_count):
        node_id, x, y, z, radius, left, right, member_offset, member_count = (
            VISIBILITY_NODE_RECORD.unpack_from(payload, offset)
        )
        offset += VISIBILITY_NODE_RECORD.size
        raw_nodes.append((node_id, (x, y, z), radius, left, right, member_offset, member_count))

    boundary_ids = (
        struct.unpack_from(f"<{boundary_count}I", payload, offset) if boundary_count else ()
    )
    offset += boundary_count * 4
    leaf_members = (
        struct.unpack_from(f"<{leaf_member_count}I", payload, offset)
        if leaf_member_count
        else ()
    )

    chunks: list[ChunkBounds] = []
    for chunk_id, center, radius, member_offset, member_count in raw_chunks:
        end = member_offset + member_count
        if end > len(boundary_ids):
            raise ChunkVisibilityError(f"Chunk {chunk_id} boundary range is invalid")
        chunks.append(
            ChunkBounds(
                chunk_id=chunk_id,
                center=center,
                angular_radius=radius,
                boundary_triangle_ids=tuple(boundary_ids[member_offset:end]),
            )
        )

    nodes: list[VisibilityNode] = []
    for node_id, center, radius, left, right, member_offset, member_count in raw_nodes:
        end = member_offset + member_count
        if end > len(leaf_members):
            raise ChunkVisibilityError(f"Visibility node {node_id} member range is invalid")
        nodes.append(
            VisibilityNode(
                node_id=node_id,
                center=center,
                angular_radius=radius,
                left=left,
                right=right,
                chunk_ids=tuple(leaf_members[member_offset:end]),
            )
        )

    topology_hash = topology_hash_raw.hex()
    layout_hash = layout_hash_raw.hex()
    if int(manifest.get("frequency", -1)) != frequency:
        raise ChunkVisibilityError("Chunk visibility manifest frequency mismatch")
    if str(manifest.get("topologyHash", "")) != topology_hash:
        raise ChunkVisibilityError("Chunk visibility manifest topology hash mismatch")
    if str(manifest.get("layoutHash", "")) != layout_hash:
        raise ChunkVisibilityError("Chunk visibility manifest layout hash mismatch")
    chunks_tuple = tuple(chunks)
    nodes_tuple = tuple(nodes)
    stable_hash = _stable_visibility_hash(
        frequency,
        topology_hash,
        layout_hash,
        leaf_size,
        chunks_tuple,
        nodes_tuple,
        root_id,
    )
    if stable_hash != str(manifest.get("visibilityHash", "")):
        raise ChunkVisibilityError("Chunk visibility stable hash mismatch")
    return ChunkVisibilityIndex(
        frequency=frequency,
        topology_hash=topology_hash,
        layout_hash=layout_hash,
        leaf_size=leaf_size,
        chunks=chunks_tuple,
        nodes=nodes_tuple,
        root_id=root_id,
        stable_hash=stable_hash,
    )


def verify_chunk_visibility_cache(directory: str | Path) -> dict:
    index = load_chunk_visibility_cache(directory)
    validation = index.validate()
    return {
        "valid": validation.valid,
        "frequency": index.frequency,
        "chunks": validation.chunk_count,
        "nodes": validation.node_count,
        "leaves": validation.leaf_count,
        "boundaryTriangleReferences": validation.boundary_point_count,
        "visibilityHash": index.stable_hash,
        "issues": list(validation.issues),
    }


def _cap_intersects_viewport(
    center: tuple[float, float, float],
    angular_radius: float,
    yaw: float,
    pitch: float,
    sphere_radius: float,
    screen_center_x: float,
    screen_center_y: float,
    width: int,
    height: int,
    margin: float,
) -> bool:
    x, y, z = _rotate_point(center, yaw, pitch)
    if angular_radius < math.pi / 2.0 and z < -math.sin(angular_radius) - 1e-12:
        return False
    if angular_radius >= math.pi / 2.0:
        projected_bound = sphere_radius * 2.0
    else:
        projected_bound = sphere_radius * 2.0 * math.sin(angular_radius / 2.0)
    projected_x = screen_center_x + x * sphere_radius
    projected_y = screen_center_y - y * sphere_radius
    return not (
        projected_x + projected_bound < -margin
        or projected_x - projected_bound > width + margin
        or projected_y + projected_bound < -margin
        or projected_y - projected_bound > height + margin
    )


def _cap_from_points(
    points: Iterable[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], float]:
    values = tuple(points)
    if not values:
        raise ChunkVisibilityError("Cannot build a spherical cap without points")
    summed = (
        sum(point[0] for point in values),
        sum(point[1] for point in values),
        sum(point[2] for point in values),
    )
    try:
        center = _normalize(summed)
    except ChunkVisibilityError:
        center = values[0]
    radius = max(_angle(center, point) for point in values) + 1e-12
    return center, min(math.pi, radius)


def _cap_from_bounds(
    chunks: list[ChunkBounds], chunk_ids: tuple[int, ...]
) -> tuple[tuple[float, float, float], float]:
    summed = (
        sum(chunks[chunk_id].center[0] for chunk_id in chunk_ids),
        sum(chunks[chunk_id].center[1] for chunk_id in chunk_ids),
        sum(chunks[chunk_id].center[2] for chunk_id in chunk_ids),
    )
    try:
        center = _normalize(summed)
    except ChunkVisibilityError:
        center = chunks[min(chunk_ids)].center
    radius = 0.0
    for chunk_id in chunk_ids:
        bound = chunks[chunk_id]
        radius = max(radius, _angle(center, bound.center) + bound.angular_radius)
    return center, min(math.pi, radius + 1e-12)


def _widest_axis(chunks: list[ChunkBounds], chunk_ids: tuple[int, ...]) -> int:
    ranges = []
    for axis in range(3):
        values = [chunks[chunk_id].center[axis] for chunk_id in chunk_ids]
        ranges.append(max(values) - min(values))
    return max(range(3), key=lambda axis: (ranges[axis], -axis))


def _cap_contains(
    outer_center: tuple[float, float, float],
    outer_radius: float,
    inner_center: tuple[float, float, float],
    inner_radius: float,
) -> bool:
    if outer_radius >= math.pi - 1e-9:
        return True
    return _angle(outer_center, inner_center) + inner_radius <= outer_radius + 1e-8


def _stable_visibility_hash(
    frequency: int,
    topology_hash: str,
    layout_hash: str,
    leaf_size: int,
    chunks: tuple[ChunkBounds, ...],
    nodes: tuple[VisibilityNode, ...],
    root_id: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<IIII", VISIBILITY_VERSION, frequency, leaf_size, root_id))
    digest.update(bytes.fromhex(topology_hash))
    digest.update(bytes.fromhex(layout_hash))
    for chunk in chunks:
        digest.update(
            struct.pack(
                "<I4dI",
                chunk.chunk_id,
                chunk.center[0],
                chunk.center[1],
                chunk.center[2],
                chunk.angular_radius,
                len(chunk.boundary_triangle_ids),
            )
        )
        for triangle_id in chunk.boundary_triangle_ids:
            digest.update(struct.pack("<I", triangle_id))
    for node in nodes:
        digest.update(
            struct.pack(
                "<I4diiI",
                node.node_id,
                node.center[0],
                node.center[1],
                node.center[2],
                node.angular_radius,
                node.left,
                node.right,
                len(node.chunk_ids),
            )
        )
        for chunk_id in node.chunk_ids:
            digest.update(struct.pack("<I", chunk_id))
    return digest.hexdigest()


def _rotate_point(
    point: tuple[float, float, float], yaw: float, pitch: float
) -> tuple[float, float, float]:
    x, y, z = point
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    x, z = x * cosine_yaw + z * sine_yaw, -x * sine_yaw + z * cosine_yaw
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    y, z = y * cosine_pitch - z * sine_pitch, y * sine_pitch + z * cosine_pitch
    return x, y, z


def _angle(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return math.acos(max(-1.0, min(1.0, _dot(first, second))))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-15:
        raise ChunkVisibilityError("Cannot normalize a zero vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


def _dot(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _unit_vector(vector: tuple[float, float, float]) -> bool:
    length = math.sqrt(_dot(vector, vector))
    return abs(length - 1.0) <= 1e-7


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
