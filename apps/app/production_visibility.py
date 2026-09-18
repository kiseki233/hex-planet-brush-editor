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

from .chunk_layout import ChunkLayoutError
from .production_layout import ProductionChunkLayout

PRODUCTION_VISIBILITY_FORMAT = "HEX_PLANET_PRODUCTION_VISIBILITY"
PRODUCTION_VISIBILITY_VERSION = 1
PRODUCTION_VISIBILITY_MAGIC = b"HPPVIS1\0"
PRODUCTION_VISIBILITY_HEADER = struct.Struct("<8s6I32s32s32s")
PRODUCTION_CHUNK_BOUND = struct.Struct("<I4dI")
PRODUCTION_VISIBILITY_NODE = struct.Struct("<I4diiII")
DEFAULT_PRODUCTION_LEAF_SIZE = 12


class ProductionVisibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ProductionChunkBound:
    chunk_id: int
    center: tuple[float, float, float]
    angular_radius: float
    seed_cell_id: int


@dataclass(frozen=True)
class ProductionVisibilityNode:
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
class ProductionVisibilityQuery:
    chunk_ids: tuple[int, ...]
    visited_nodes: int
    rejected_nodes: int
    tested_chunks: int
    candidate_cells: int


@dataclass(frozen=True)
class ProductionVisibilityValidation:
    valid: bool
    chunk_count: int
    node_count: int
    leaf_count: int
    stable_hash: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ProductionVisibilityCacheInfo:
    directory: Path
    index_path: Path
    manifest_path: Path
    chunk_count: int
    node_count: int
    byte_size: int
    crc32: int
    sha256: str


@dataclass(frozen=True)
class ProductionVisibilityIndex:
    frequency: int
    topology_hash: str
    layout_hash: str
    leaf_size: int
    chunks: tuple[ProductionChunkBound, ...]
    nodes: tuple[ProductionVisibilityNode, ...]
    root_id: int
    stable_hash: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def query(
        self,
        layout: ProductionChunkLayout,
        yaw: float,
        pitch: float,
        zoom: float,
        width: int,
        height: int,
        margin: float = 18.0,
    ) -> ProductionVisibilityQuery:
        if layout.stable_hash != self.layout_hash:
            raise ProductionVisibilityError("Production layout hash mismatch")
        width = max(1, int(width))
        height = max(1, int(height))
        base_radius = max(20.0, min(width, height) * 0.43)
        sphere_radius = base_radius * max(0.8, float(zoom))
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
        return ProductionVisibilityQuery(
            chunk_ids=chunk_ids,
            visited_nodes=visited_nodes,
            rejected_nodes=rejected_nodes,
            tested_chunks=tested_chunks,
            candidate_cells=sum(layout.records[chunk_id].cell_count for chunk_id in chunk_ids),
        )

    def validate(
        self, layout: ProductionChunkLayout | None = None
    ) -> ProductionVisibilityValidation:
        issues: list[str] = []
        if self.leaf_size < 1:
            issues.append("leaf size must be positive")
        if not self.nodes:
            issues.append("visibility hierarchy has no nodes")
        if self.root_id < 0 or self.root_id >= len(self.nodes):
            issues.append("root node id is invalid")
        if any(bound.chunk_id != index for index, bound in enumerate(self.chunks)):
            issues.append("chunk bounds ids are not contiguous")
        if any(node.node_id != index for index, node in enumerate(self.nodes)):
            issues.append("node ids are not contiguous")
        if layout is not None:
            if layout.frequency != self.frequency:
                issues.append("layout frequency mismatch")
            if layout.stable_hash != self.layout_hash:
                issues.append("layout hash mismatch")
            if layout.topology_hash != self.topology_hash:
                issues.append("topology hash mismatch")
            if layout.chunk_count != self.chunk_count:
                issues.append("chunk count mismatch")

        seen_nodes: set[int] = set()
        seen_chunks: list[int] = []
        visiting: set[int] = set()

        def walk(node_id: int) -> None:
            if node_id in visiting:
                issues.append("visibility hierarchy contains a cycle")
                return
            if node_id in seen_nodes:
                issues.append(f"node {node_id} has multiple parents")
                return
            if node_id < 0 or node_id >= len(self.nodes):
                issues.append(f"invalid node reference {node_id}")
                return
            visiting.add(node_id)
            seen_nodes.add(node_id)
            node = self.nodes[node_id]
            if not _is_unit(node.center):
                issues.append(f"node {node_id} center is not normalized")
            if node.angular_radius < 0.0 or node.angular_radius > math.pi + 1e-9:
                issues.append(f"node {node_id} angular radius is invalid")
            if node.is_leaf:
                if not node.chunk_ids:
                    issues.append(f"leaf {node_id} is empty")
                if len(node.chunk_ids) > self.leaf_size:
                    issues.append(f"leaf {node_id} exceeds leaf size")
                for chunk_id in node.chunk_ids:
                    if chunk_id < 0 or chunk_id >= self.chunk_count:
                        issues.append(f"leaf {node_id} references invalid chunk {chunk_id}")
                        continue
                    seen_chunks.append(chunk_id)
                    bound = self.chunks[chunk_id]
                    if not _cap_contains(
                        node.center,
                        node.angular_radius,
                        bound.center,
                        bound.angular_radius,
                    ):
                        issues.append(f"leaf {node_id} does not contain chunk {chunk_id}")
            else:
                if node.chunk_ids:
                    issues.append(f"internal node {node_id} stores chunk ids")
                if node.left < 0 or node.right < 0:
                    issues.append(f"internal node {node_id} is missing a child")
                if node.left >= 0:
                    walk(node.left)
                if node.right >= 0:
                    walk(node.right)
            visiting.remove(node_id)

        if 0 <= self.root_id < len(self.nodes):
            walk(self.root_id)
        if len(seen_nodes) != len(self.nodes):
            issues.append("visibility hierarchy contains unused nodes")
        if sorted(seen_chunks) != list(range(self.chunk_count)):
            issues.append("visibility hierarchy does not cover every chunk exactly once")
        for bound in self.chunks:
            if not _is_unit(bound.center):
                issues.append(f"chunk {bound.chunk_id} center is not normalized")
            if bound.angular_radius < 0.0 or bound.angular_radius > math.pi + 1e-9:
                issues.append(f"chunk {bound.chunk_id} angular radius is invalid")
            if layout is not None:
                address = layout.topology.owner_address(bound.seed_cell_id)
                key = (
                    address.face_id,
                    layout._tile_index(address.weight_b),
                    layout._tile_index(address.weight_c),
                )
                if layout._key_to_chunk.get(key) != bound.chunk_id:
                    issues.append(f"chunk {bound.chunk_id} seed CellId is not a member")
        expected_hash = _stable_hash(
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
        return ProductionVisibilityValidation(
            valid=not issues,
            chunk_count=self.chunk_count,
            node_count=self.node_count,
            leaf_count=sum(node.is_leaf for node in self.nodes),
            stable_hash=self.stable_hash,
            issues=tuple(issues),
        )


def build_production_visibility_index(
    layout: ProductionChunkLayout,
    leaf_size: int = DEFAULT_PRODUCTION_LEAF_SIZE,
) -> ProductionVisibilityIndex:
    if leaf_size < 1 or leaf_size > 1024:
        raise ProductionVisibilityError("Production visibility leaf size must be between 1 and 1024")
    topology = layout.topology
    chunks: list[ProductionChunkBound] = []
    cell_margin = min(math.pi, 4.0 / max(1, layout.frequency))
    for record in layout.records:
        sample_ids: list[int] = []
        for tile_b, tile_c in record.tile_keys:
            for weight_b, weight_c in _tile_sample_weights(layout, tile_b, tile_c):
                weight_a = layout.frequency - weight_b - weight_c
                sample_ids.append(
                    topology.point_id(record.base_face, weight_a, weight_b, weight_c)
                )
        if not sample_ids:
            raise ProductionVisibilityError(f"Production chunk {record.chunk_id} has no samples")
        sample_ids = list(dict.fromkeys(sample_ids))
        points = [topology.cell_center(cell_id) for cell_id in sample_ids]
        center = _normalize(
            (
                sum(point[0] for point in points),
                sum(point[1] for point in points),
                sum(point[2] for point in points),
            )
        )
        radius = min(
            math.pi,
            max(_angle(center, point) for point in points) + cell_margin + 1e-12,
        )
        seed = next(
            cell_id
            for tile_b, tile_c in record.tile_keys
            for cell_id in layout._iter_tile_cells(record.base_face, tile_b, tile_c)
        )
        chunks.append(
            ProductionChunkBound(
                chunk_id=record.chunk_id,
                center=center,
                angular_radius=radius,
                seed_cell_id=seed,
            )
        )

    nodes: list[ProductionVisibilityNode | None] = []

    def build_node(chunk_ids: tuple[int, ...]) -> int:
        node_id = len(nodes)
        nodes.append(None)
        center, radius = _cap_from_bounds(chunks, chunk_ids)
        if len(chunk_ids) <= leaf_size:
            nodes[node_id] = ProductionVisibilityNode(
                node_id=node_id,
                center=center,
                angular_radius=radius,
                left=-1,
                right=-1,
                chunk_ids=tuple(sorted(chunk_ids)),
            )
            return node_id
        axis = _widest_axis(chunks, chunk_ids)
        ordered = tuple(sorted(chunk_ids, key=lambda item: (chunks[item].center[axis], item)))
        midpoint = len(ordered) // 2
        left = build_node(ordered[:midpoint])
        right = build_node(ordered[midpoint:])
        nodes[node_id] = ProductionVisibilityNode(
            node_id=node_id,
            center=center,
            angular_radius=radius,
            left=left,
            right=right,
            chunk_ids=(),
        )
        return node_id

    root_id = build_node(tuple(range(layout.chunk_count)))
    final_chunks = tuple(chunks)
    final_nodes = tuple(node for node in nodes if node is not None)
    stable_hash = _stable_hash(
        layout.frequency,
        layout.topology_hash,
        layout.stable_hash,
        leaf_size,
        final_chunks,
        final_nodes,
        root_id,
    )
    index = ProductionVisibilityIndex(
        frequency=layout.frequency,
        topology_hash=layout.topology_hash,
        layout_hash=layout.stable_hash,
        leaf_size=leaf_size,
        chunks=final_chunks,
        nodes=final_nodes,
        root_id=root_id,
        stable_hash=stable_hash,
    )
    validation = index.validate(layout)
    if not validation.valid:
        raise ProductionVisibilityError(
            "Production visibility validation failed: " + "; ".join(validation.issues[:8])
        )
    return index


def write_production_visibility_cache(
    index: ProductionVisibilityIndex,
    topology_root: str | Path,
) -> ProductionVisibilityCacheInfo:
    directory = Path(topology_root) / f"ico_dual_f{index.frequency}_v2"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "production_visibility.idx"
    manifest_path = directory / "production_visibility.json"

    leaf_members: list[int] = []
    node_payload = bytearray()
    for node in index.nodes:
        offset = len(leaf_members)
        leaf_members.extend(node.chunk_ids)
        node_payload.extend(
            PRODUCTION_VISIBILITY_NODE.pack(
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
    chunk_payload = b"".join(
        PRODUCTION_CHUNK_BOUND.pack(
            bound.chunk_id,
            bound.center[0],
            bound.center[1],
            bound.center[2],
            bound.angular_radius,
            bound.seed_cell_id,
        )
        for bound in index.chunks
    )
    member_payload = b"".join(struct.pack("<I", chunk_id) for chunk_id in leaf_members)
    header = PRODUCTION_VISIBILITY_HEADER.pack(
        PRODUCTION_VISIBILITY_MAGIC,
        PRODUCTION_VISIBILITY_VERSION,
        index.frequency,
        index.chunk_count,
        index.node_count,
        index.root_id,
        index.leaf_size,
        bytes.fromhex(index.topology_hash),
        bytes.fromhex(index.layout_hash),
        bytes.fromhex(index.stable_hash),
    )
    body = header + chunk_payload + bytes(node_payload) + member_payload
    crc32 = zlib.crc32(body) & 0xFFFFFFFF
    payload = body + struct.pack("<I", crc32)
    sha256 = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(index_path, payload)
    _atomic_write_json(
        manifest_path,
        {
            "format": PRODUCTION_VISIBILITY_FORMAT,
            "version": PRODUCTION_VISIBILITY_VERSION,
            "frequency": index.frequency,
            "topologyHash": index.topology_hash,
            "layoutHash": index.layout_hash,
            "leafSize": index.leaf_size,
            "chunkCount": index.chunk_count,
            "nodeCount": index.node_count,
            "rootId": index.root_id,
            "stableHash": index.stable_hash,
            "indexFile": index_path.name,
            "indexBytes": len(payload),
            "crc32": f"{crc32:08x}",
            "sha256": sha256,
        },
    )
    return ProductionVisibilityCacheInfo(
        directory=directory,
        index_path=index_path,
        manifest_path=manifest_path,
        chunk_count=index.chunk_count,
        node_count=index.node_count,
        byte_size=len(payload),
        crc32=crc32,
        sha256=sha256,
    )


def load_production_visibility_cache(
    directory: str | Path,
    layout: ProductionChunkLayout,
) -> ProductionVisibilityIndex:
    directory = Path(directory)
    manifest = json.loads((directory / "production_visibility.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PRODUCTION_VISIBILITY_FORMAT:
        raise ProductionVisibilityError("Unsupported production visibility manifest")
    index_path = directory / str(manifest.get("indexFile", "production_visibility.idx"))
    payload = index_path.read_bytes()
    if len(payload) < PRODUCTION_VISIBILITY_HEADER.size + 4:
        raise ProductionVisibilityError("Production visibility index is truncated")
    body, crc_payload = payload[:-4], payload[-4:]
    expected_crc = struct.unpack("<I", crc_payload)[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProductionVisibilityError("Production visibility CRC mismatch")
    if hashlib.sha256(payload).hexdigest() != manifest.get("sha256"):
        raise ProductionVisibilityError("Production visibility SHA-256 mismatch")
    (
        magic,
        version,
        frequency,
        chunk_count,
        node_count,
        root_id,
        leaf_size,
        topology_hash_bytes,
        layout_hash_bytes,
        stable_hash_bytes,
    ) = PRODUCTION_VISIBILITY_HEADER.unpack_from(body, 0)
    if magic != PRODUCTION_VISIBILITY_MAGIC or version != PRODUCTION_VISIBILITY_VERSION:
        raise ProductionVisibilityError("Unsupported production visibility index version")
    if frequency != layout.frequency or chunk_count != layout.chunk_count:
        raise ProductionVisibilityError("Production visibility identity mismatch")
    topology_hash = topology_hash_bytes.hex()
    layout_hash = layout_hash_bytes.hex()
    stable_hash = stable_hash_bytes.hex()
    if topology_hash != layout.topology_hash or layout_hash != layout.stable_hash:
        raise ProductionVisibilityError("Production visibility hash mismatch")

    offset = PRODUCTION_VISIBILITY_HEADER.size
    chunks: list[ProductionChunkBound] = []
    for _ in range(chunk_count):
        if offset + PRODUCTION_CHUNK_BOUND.size > len(body):
            raise ProductionVisibilityError("Production chunk bounds are truncated")
        chunk_id, x, y, z, radius, seed = PRODUCTION_CHUNK_BOUND.unpack_from(body, offset)
        offset += PRODUCTION_CHUNK_BOUND.size
        chunks.append(ProductionChunkBound(chunk_id, (x, y, z), radius, seed))
    raw_nodes: list[tuple[int, tuple[float, float, float], float, int, int, int, int]] = []
    maximum_member = 0
    for _ in range(node_count):
        if offset + PRODUCTION_VISIBILITY_NODE.size > len(body):
            raise ProductionVisibilityError("Production visibility nodes are truncated")
        node_id, x, y, z, radius, left, right, member_offset, member_count = (
            PRODUCTION_VISIBILITY_NODE.unpack_from(body, offset)
        )
        offset += PRODUCTION_VISIBILITY_NODE.size
        maximum_member = max(maximum_member, member_offset + member_count)
        raw_nodes.append((node_id, (x, y, z), radius, left, right, member_offset, member_count))
    expected_size = offset + maximum_member * 4
    if expected_size != len(body):
        raise ProductionVisibilityError(
            f"Production visibility size mismatch: expected {expected_size}, received {len(body)}"
        )
    members = struct.unpack_from(f"<{maximum_member}I", body, offset) if maximum_member else ()
    nodes = tuple(
        ProductionVisibilityNode(
            node_id=node_id,
            center=center,
            angular_radius=radius,
            left=left,
            right=right,
            chunk_ids=tuple(members[member_offset : member_offset + member_count]),
        )
        for node_id, center, radius, left, right, member_offset, member_count in raw_nodes
    )
    index = ProductionVisibilityIndex(
        frequency=frequency,
        topology_hash=topology_hash,
        layout_hash=layout_hash,
        leaf_size=leaf_size,
        chunks=tuple(chunks),
        nodes=nodes,
        root_id=root_id,
        stable_hash=stable_hash,
    )
    validation = index.validate(layout)
    if not validation.valid:
        raise ProductionVisibilityError(
            "Production visibility cache validation failed: " + "; ".join(validation.issues[:8])
        )
    if stable_hash != manifest.get("stableHash"):
        raise ProductionVisibilityError("Production visibility manifest stable hash mismatch")
    return index


def _tile_sample_weights(
    layout: ProductionChunkLayout, tile_b: int, tile_c: int
) -> tuple[tuple[int, int], ...]:
    b0, b1 = layout._tile_bounds(tile_b)
    c0, c1 = layout._tile_bounds(tile_c)
    frequency = layout.frequency
    candidates: set[tuple[int, int]] = set()

    def add(weight_b: int, weight_c: int) -> None:
        if b0 <= weight_b <= b1 and c0 <= weight_c <= c1 and weight_b + weight_c <= frequency:
            candidates.add((weight_b, weight_c))

    for weight_b in (b0, b1):
        add(weight_b, c0)
        add(weight_b, min(c1, frequency - weight_b))
    for weight_c in (c0, c1):
        add(b0, weight_c)
        add(min(b1, frequency - weight_c), weight_c)
    add((b0 + b1) // 2, (c0 + c1) // 2)
    if not candidates:
        for weight_b in range(b0, b1 + 1):
            maximum_c = min(c1, frequency - weight_b)
            if maximum_c >= c0:
                add(weight_b, c0)
                add(weight_b, maximum_c)
    return tuple(sorted(candidates))


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
    projected_bound = (
        sphere_radius * 2.0
        if angular_radius >= math.pi / 2.0
        else sphere_radius * 2.0 * math.sin(angular_radius / 2.0)
    )
    projected_x = screen_center_x + x * sphere_radius
    projected_y = screen_center_y - y * sphere_radius
    return not (
        projected_x + projected_bound < -margin
        or projected_x - projected_bound > width + margin
        or projected_y + projected_bound < -margin
        or projected_y - projected_bound > height + margin
    )


def _cap_from_bounds(
    chunks: list[ProductionChunkBound], chunk_ids: tuple[int, ...]
) -> tuple[tuple[float, float, float], float]:
    summed = (
        sum(chunks[chunk_id].center[0] for chunk_id in chunk_ids),
        sum(chunks[chunk_id].center[1] for chunk_id in chunk_ids),
        sum(chunks[chunk_id].center[2] for chunk_id in chunk_ids),
    )
    try:
        center = _normalize(summed)
    except ProductionVisibilityError:
        center = chunks[min(chunk_ids)].center
    radius = max(
        _angle(center, chunks[chunk_id].center) + chunks[chunk_id].angular_radius
        for chunk_id in chunk_ids
    )
    return center, min(math.pi, radius + 1e-12)


def _widest_axis(chunks: list[ProductionChunkBound], chunk_ids: tuple[int, ...]) -> int:
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
    return outer_radius >= math.pi - 1e-9 or (
        _angle(outer_center, inner_center) + inner_radius <= outer_radius + 1e-8
    )


def _stable_hash(
    frequency: int,
    topology_hash: str,
    layout_hash: str,
    leaf_size: int,
    chunks: tuple[ProductionChunkBound, ...],
    nodes: tuple[ProductionVisibilityNode, ...],
    root_id: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        struct.pack(
            "<IIIII",
            PRODUCTION_VISIBILITY_VERSION,
            frequency,
            leaf_size,
            root_id,
            len(chunks),
        )
    )
    digest.update(bytes.fromhex(topology_hash))
    digest.update(bytes.fromhex(layout_hash))
    for bound in chunks:
        digest.update(
            PRODUCTION_CHUNK_BOUND.pack(
                bound.chunk_id,
                bound.center[0],
                bound.center[1],
                bound.center[2],
                bound.angular_radius,
                bound.seed_cell_id,
            )
        )
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
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    return x, y, z


def _angle(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return math.acos(max(-1.0, min(1.0, _dot(first, second))))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length <= 1e-15:
        raise ProductionVisibilityError("Cannot normalize a zero vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


def _dot(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _is_unit(vector: tuple[float, float, float]) -> bool:
    return abs(math.sqrt(_dot(vector, vector)) - 1.0) <= 1e-7


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
