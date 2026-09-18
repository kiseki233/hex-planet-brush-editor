from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .topology import BASE_FACES, BASE_VERTICES, DualTopology

LAYOUT_FORMAT = "HEX_PLANET_CHUNK_LAYOUT"
LAYOUT_VERSION = 1
LAYOUT_MAGIC = b"HPCHNK1\0"
LAYOUT_HEADER = struct.Struct("<8sIIIIII32s")
CHUNK_RECORD = struct.Struct("<HHII")
DEFAULT_TARGET_CELLS = 256


class ChunkLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkDefinition:
    chunk_id: int
    base_face: int
    cell_ids: tuple[int, ...]


@dataclass(frozen=True)
class ChunkLayoutValidation:
    valid: bool
    cell_count: int
    chunk_count: int
    target_cells: int
    smallest_chunk: int
    largest_chunk: int
    connected_chunks: bool
    stable_hash: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ChunkLayout:
    frequency: int
    topology_hash: str
    target_cells: int
    chunks: tuple[ChunkDefinition, ...]
    cell_to_chunk: tuple[int, ...]
    cell_to_local_index: tuple[int, ...]
    stable_hash: str

    @property
    def cell_count(self) -> int:
        return len(self.cell_to_chunk)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def chunk_for_cell(self, cell_id: int) -> tuple[int, int]:
        if cell_id < 0 or cell_id >= self.cell_count:
            raise IndexError(f"CellId is outside the layout: {cell_id}")
        return self.cell_to_chunk[cell_id], self.cell_to_local_index[cell_id]

    def validate(self, topology: DualTopology | None = None) -> ChunkLayoutValidation:
        issues: list[str] = []
        if self.target_cells < 1:
            issues.append("target cell count must be positive")
        if len(self.cell_to_local_index) != self.cell_count:
            issues.append("cell-to-local-index length mismatch")

        seen = [False] * self.cell_count
        connected = True
        for expected_chunk_id, chunk in enumerate(self.chunks):
            if chunk.chunk_id != expected_chunk_id:
                issues.append(
                    f"chunk id mismatch: expected {expected_chunk_id}, received {chunk.chunk_id}"
                )
            if chunk.base_face < 0 or chunk.base_face >= len(BASE_FACES):
                issues.append(f"chunk {chunk.chunk_id} has invalid base face {chunk.base_face}")
            if not chunk.cell_ids:
                issues.append(f"chunk {chunk.chunk_id} is empty")
            if len(chunk.cell_ids) > self.target_cells:
                issues.append(
                    f"chunk {chunk.chunk_id} exceeds target: {len(chunk.cell_ids)} > {self.target_cells}"
                )
            for local_index, cell_id in enumerate(chunk.cell_ids):
                if cell_id < 0 or cell_id >= self.cell_count:
                    issues.append(f"chunk {chunk.chunk_id} references invalid CellId {cell_id}")
                    continue
                if seen[cell_id]:
                    issues.append(f"CellId {cell_id} appears in more than one chunk")
                seen[cell_id] = True
                if self.cell_to_chunk[cell_id] != chunk.chunk_id:
                    issues.append(f"CellId {cell_id} has inconsistent chunk mapping")
                if self.cell_to_local_index[cell_id] != local_index:
                    issues.append(f"CellId {cell_id} has inconsistent local index")

            if topology is not None and chunk.cell_ids:
                members = set(chunk.cell_ids)
                reached = {chunk.cell_ids[0]}
                queue = deque([chunk.cell_ids[0]])
                while queue:
                    current = queue.popleft()
                    for neighbor in topology.neighbors[current]:
                        if neighbor in members and neighbor not in reached:
                            reached.add(neighbor)
                            queue.append(neighbor)
                if len(reached) != len(members):
                    connected = False
                    issues.append(f"chunk {chunk.chunk_id} is not connected")

        missing = [cell_id for cell_id, value in enumerate(seen) if not value]
        if missing:
            issues.append(f"layout does not cover {len(missing)} cells")
        if topology is not None:
            if topology.frequency != self.frequency:
                issues.append("topology frequency mismatch")
            if topology.stable_hash != self.topology_hash:
                issues.append("topology hash mismatch")
            if topology.cell_count != self.cell_count:
                issues.append("topology cell count mismatch")

        sizes = [len(chunk.cell_ids) for chunk in self.chunks]
        return ChunkLayoutValidation(
            valid=not issues,
            cell_count=self.cell_count,
            chunk_count=self.chunk_count,
            target_cells=self.target_cells,
            smallest_chunk=min(sizes, default=0),
            largest_chunk=max(sizes, default=0),
            connected_chunks=connected,
            stable_hash=self.stable_hash,
            issues=tuple(issues),
        )


@dataclass(frozen=True)
class ChunkLayoutCacheInfo:
    directory: Path
    manifest_path: Path
    index_path: Path
    crc32: str
    sha256: str


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if length == 0.0:
        raise ChunkLayoutError("Cannot normalize a zero vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _base_face_centers() -> tuple[tuple[float, float, float], ...]:
    centers: list[tuple[float, float, float]] = []
    for first, second, third in BASE_FACES:
        a, b, c = BASE_VERTICES[first], BASE_VERTICES[second], BASE_VERTICES[third]
        centers.append(_normalize((a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])))
    return tuple(centers)


FACE_CENTERS = _base_face_centers()


def _assign_base_faces(topology: DualTopology) -> list[int]:
    assignments: list[int] = []
    for center in topology.cell_centers:
        best_face = 0
        best_score = _dot(center, FACE_CENTERS[0])
        for face_id in range(1, len(FACE_CENTERS)):
            score = _dot(center, FACE_CENTERS[face_id])
            if score > best_score + 1e-15:
                best_face = face_id
                best_score = score
        assignments.append(best_face)
    return assignments


def _stable_layout_hash(
    frequency: int,
    topology_hash: str,
    target_cells: int,
    chunks: list[ChunkDefinition],
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<III", LAYOUT_VERSION, frequency, target_cells))
    digest.update(bytes.fromhex(topology_hash))
    for chunk in chunks:
        digest.update(struct.pack("<IHI", chunk.chunk_id, chunk.base_face, len(chunk.cell_ids)))
        for cell_id in chunk.cell_ids:
            digest.update(struct.pack("<I", cell_id))
    return digest.hexdigest()



def _subset_is_connected(
    members: set[int], neighbors: tuple[tuple[int, ...], ...]
) -> bool:
    if not members:
        return False
    seed = min(members)
    reached = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if neighbor in members and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return len(reached) == len(members)


def _split_connected_balanced(
    members: set[int],
    neighbors: tuple[tuple[int, ...], ...],
    target_cells: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total = len(members)
    lower = max(1, total - target_cells)
    upper = min(target_cells, total - 1)
    desired = total // 2
    best: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None

    for seed in sorted(members):
        selected = {seed}
        remaining = set(members)
        remaining.remove(seed)
        if remaining and not _subset_is_connected(remaining, neighbors):
            continue

        while True:
            selected_count = len(selected)
            if lower <= selected_count <= upper and remaining:
                score = abs(selected_count - desired)
                candidate = (score, tuple(sorted(selected)), tuple(sorted(remaining)))
                if best is None or candidate < best:
                    best = candidate
                if score == 0:
                    return candidate[1], candidate[2]
            if selected_count >= upper:
                break

            frontier = {
                neighbor
                for cell_id in selected
                for neighbor in neighbors[cell_id]
                if neighbor in remaining
            }
            safe_candidates: list[tuple[int, int]] = []
            for candidate_id in frontier:
                new_remaining = remaining - {candidate_id}
                if new_remaining and not _subset_is_connected(new_remaining, neighbors):
                    continue
                shared_edges = sum(1 for neighbor in neighbors[candidate_id] if neighbor in selected)
                safe_candidates.append((-shared_edges, candidate_id))
            if not safe_candidates:
                break
            _, chosen = min(safe_candidates)
            selected.add(chosen)
            remaining.remove(chosen)

    if best is None:
        raise ChunkLayoutError("Cannot rebalance the final connected chunk pair")
    return best[1], best[2]


def _shared_edge_count(
    first: set[int],
    second: set[int],
    neighbors: tuple[tuple[int, ...], ...],
) -> int:
    return sum(1 for cell_id in first for neighbor in neighbors[cell_id] if neighbor in second)


def _rebalance_face_chunks(
    face_chunks: list[tuple[int, ...]],
    neighbors: tuple[tuple[int, ...], ...],
    target_cells: int,
) -> list[tuple[int, ...]]:
    minimum_preferred = max(1, target_cells // 2)
    while True:
        small_indices = [
            index
            for index, members in enumerate(face_chunks)
            if len(members) < minimum_preferred
        ]
        changed = False
        for small_index in small_indices:
            if small_index >= len(face_chunks):
                continue
            small = set(face_chunks[small_index])
            candidates: list[tuple[int, int, int]] = []
            for other_index, other_members in enumerate(face_chunks):
                if other_index == small_index:
                    continue
                other = set(other_members)
                shared = _shared_edge_count(small, other, neighbors)
                if shared > 0:
                    candidates.append((-shared, len(small) + len(other), other_index))
            if not candidates:
                continue

            _, _, other_index = min(candidates)
            combined = small | set(face_chunks[other_index])
            if len(combined) <= target_cells:
                replacement = [tuple(sorted(combined))]
            else:
                first, second = _split_connected_balanced(combined, neighbors, target_cells)
                replacement = [first, second]

            low, high = sorted((small_index, other_index))
            face_chunks = [
                members
                for index, members in enumerate(face_chunks)
                if index not in (low, high)
            ]
            face_chunks[low:low] = replacement
            changed = True
            break
        if not changed:
            return face_chunks


def build_chunk_layout(
    topology: DualTopology,
    target_cells: int = DEFAULT_TARGET_CELLS,
) -> ChunkLayout:
    if target_cells < 1 or target_cells > 65535:
        raise ChunkLayoutError("Target chunk size must be between 1 and 65535")

    face_assignments = _assign_base_faces(topology)
    chunks: list[ChunkDefinition] = []
    cell_to_chunk = [-1] * topology.cell_count
    cell_to_local = [-1] * topology.cell_count

    for face_id in range(len(BASE_FACES)):
        remaining = {cell_id for cell_id, owner in enumerate(face_assignments) if owner == face_id}
        face_chunks: list[tuple[int, ...]] = []
        while remaining:
            seed = min(remaining)
            queue = deque([seed])
            queued = {seed}
            members: list[int] = []

            while queue and len(members) < target_cells:
                cell_id = queue.popleft()
                queued.discard(cell_id)
                if cell_id not in remaining:
                    continue
                remaining.remove(cell_id)
                members.append(cell_id)
                for neighbor in sorted(topology.neighbors[cell_id]):
                    if (
                        neighbor in remaining
                        and face_assignments[neighbor] == face_id
                        and neighbor not in queued
                    ):
                        queue.append(neighbor)
                        queued.add(neighbor)

            if not members:
                raise ChunkLayoutError(f"Cannot grow a chunk from CellId {seed}")
            face_chunks.append(tuple(members))

        face_chunks = _rebalance_face_chunks(face_chunks, topology.neighbors, target_cells)
        for members in face_chunks:
            chunk_id = len(chunks)
            definition = ChunkDefinition(chunk_id=chunk_id, base_face=face_id, cell_ids=members)
            chunks.append(definition)
            for local_index, cell_id in enumerate(members):
                cell_to_chunk[cell_id] = chunk_id
                cell_to_local[cell_id] = local_index

    stable_hash = _stable_layout_hash(topology.frequency, topology.stable_hash, target_cells, chunks)
    layout = ChunkLayout(
        frequency=topology.frequency,
        topology_hash=topology.stable_hash,
        target_cells=target_cells,
        chunks=tuple(chunks),
        cell_to_chunk=tuple(cell_to_chunk),
        cell_to_local_index=tuple(cell_to_local),
        stable_hash=stable_hash,
    )
    validation = layout.validate(topology)
    if not validation.valid:
        raise ChunkLayoutError("Chunk layout validation failed: " + "; ".join(validation.issues[:8]))
    return layout


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


def write_chunk_layout_cache(layout: ChunkLayout, topology_root: str | Path) -> ChunkLayoutCacheInfo:
    directory = Path(topology_root) / f"ico_dual_f{layout.frequency}_v1"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "chunks.idx"
    manifest_path = directory / "chunks.json"

    payload = bytearray(
        LAYOUT_HEADER.pack(
            LAYOUT_MAGIC,
            LAYOUT_VERSION,
            layout.frequency,
            layout.cell_count,
            layout.chunk_count,
            layout.target_cells,
            CHUNK_RECORD.size,
            bytes.fromhex(layout.topology_hash),
        )
    )
    payload.extend(struct.pack(f"<{layout.cell_count}I", *layout.cell_to_chunk))
    payload.extend(struct.pack(f"<{layout.cell_count}H", *layout.cell_to_local_index))

    member_offset = 0
    for chunk in layout.chunks:
        payload.extend(CHUNK_RECORD.pack(chunk.base_face, 0, member_offset, len(chunk.cell_ids)))
        member_offset += len(chunk.cell_ids)
    for chunk in layout.chunks:
        payload.extend(struct.pack(f"<{len(chunk.cell_ids)}I", *chunk.cell_ids))

    raw = bytes(payload)
    crc32 = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    sha256 = hashlib.sha256(raw).hexdigest()
    _atomic_write_bytes(index_path, raw)
    _atomic_write_json(
        manifest_path,
        {
            "format": LAYOUT_FORMAT,
            "formatVersion": LAYOUT_VERSION,
            "frequency": layout.frequency,
            "topologyHash": layout.topology_hash,
            "layoutHash": layout.stable_hash,
            "targetCells": layout.target_cells,
            "counts": {"cells": layout.cell_count, "chunks": layout.chunk_count},
            "files": {
                "index": {
                    "path": "chunks.idx",
                    "bytes": len(raw),
                    "crc32": crc32,
                    "sha256": sha256,
                }
            },
            "partition": {
                "baseFaces": 20,
                "faceOwnership": "nearest_icosahedron_face_center_lowest_id_tiebreak",
                "chunkGrowth": "deterministic_breadth_first_with_connected_small_chunk_rebalance",
            },
        },
    )
    return ChunkLayoutCacheInfo(directory, manifest_path, index_path, crc32, sha256)


def load_chunk_layout_cache(directory: str | Path) -> ChunkLayout:
    cache_dir = Path(directory)
    try:
        manifest = json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))
        payload = (cache_dir / str(manifest["files"]["index"]["path"])).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ChunkLayoutError(f"Cannot read chunk layout cache: {exc}") from exc

    actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_crc != str(manifest["files"]["index"].get("crc32", "")).lower():
        raise ChunkLayoutError("Chunk layout CRC mismatch")
    if actual_sha != str(manifest["files"]["index"].get("sha256", "")).lower():
        raise ChunkLayoutError("Chunk layout SHA-256 mismatch")
    if len(payload) < LAYOUT_HEADER.size:
        raise ChunkLayoutError("Chunk layout header is truncated")

    magic, version, frequency, cell_count, chunk_count, target_cells, record_size, topology_hash_raw = (
        LAYOUT_HEADER.unpack_from(payload)
    )
    if magic != LAYOUT_MAGIC or version != LAYOUT_VERSION:
        raise ChunkLayoutError("Unsupported chunk layout format")
    if record_size != CHUNK_RECORD.size:
        raise ChunkLayoutError("Unsupported chunk record size")

    offset = LAYOUT_HEADER.size
    chunk_map_bytes = cell_count * 4
    local_map_bytes = cell_count * 2
    chunk_records_bytes = chunk_count * CHUNK_RECORD.size
    expected_size = offset + chunk_map_bytes + local_map_bytes + chunk_records_bytes + cell_count * 4
    if len(payload) != expected_size:
        raise ChunkLayoutError(
            f"Chunk layout size mismatch: expected {expected_size}, received {len(payload)}"
        )

    cell_to_chunk = struct.unpack_from(f"<{cell_count}I", payload, offset)
    offset += chunk_map_bytes
    cell_to_local = struct.unpack_from(f"<{cell_count}H", payload, offset)
    offset += local_map_bytes

    records: list[tuple[int, int, int]] = []
    for _ in range(chunk_count):
        base_face, flags, member_offset, member_count = CHUNK_RECORD.unpack_from(payload, offset)
        offset += CHUNK_RECORD.size
        if flags != 0:
            raise ChunkLayoutError("Unsupported chunk layout flags")
        records.append((base_face, member_offset, member_count))

    members = struct.unpack_from(f"<{cell_count}I", payload, offset)
    chunks: list[ChunkDefinition] = []
    for chunk_id, (base_face, member_offset, member_count) in enumerate(records):
        end = member_offset + member_count
        if end > len(members):
            raise ChunkLayoutError(f"Chunk {chunk_id} member range is invalid")
        chunks.append(ChunkDefinition(chunk_id, base_face, tuple(members[member_offset:end])))

    topology_hash = topology_hash_raw.hex()
    stable_hash = _stable_layout_hash(frequency, topology_hash, target_cells, chunks)
    if stable_hash != str(manifest.get("layoutHash", "")).lower():
        raise ChunkLayoutError("Chunk layout stable hash mismatch")

    layout = ChunkLayout(
        frequency=frequency,
        topology_hash=topology_hash,
        target_cells=target_cells,
        chunks=tuple(chunks),
        cell_to_chunk=tuple(cell_to_chunk),
        cell_to_local_index=tuple(cell_to_local),
        stable_hash=stable_hash,
    )
    validation = layout.validate()
    if not validation.valid:
        raise ChunkLayoutError("Loaded chunk layout is invalid: " + "; ".join(validation.issues[:8]))
    return layout
