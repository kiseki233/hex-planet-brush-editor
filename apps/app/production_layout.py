from __future__ import annotations

import bisect
import hashlib
import json
import os
import struct
import tempfile
import zlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, overload

from .chunk_layout import ChunkDefinition, ChunkLayoutCacheInfo, ChunkLayoutError
from .production_topology import (
    PRODUCTION_GENERATOR_VERSION,
    PRODUCTION_ORIENTATION,
    ProductionTopology,
)

PRODUCTION_LAYOUT_FORMAT = "HEX_PLANET_PROCEDURAL_CHUNK_LAYOUT"
PRODUCTION_LAYOUT_VERSION = 3
PRODUCTION_LAYOUT_MAGIC = b"HPPROD3\0"
PRODUCTION_LAYOUT_HEADER = struct.Struct("<8sIIIIII32s32s")
PRODUCTION_CHUNK_RECORD = struct.Struct("<HHIII")
PRODUCTION_TILE_KEY = struct.Struct("<HH")
DEFAULT_TILE_SIDE = 16


@dataclass(frozen=True)
class ProductionChunkRecord:
    chunk_id: int
    base_face: int
    tile_keys: tuple[tuple[int, int], ...]
    cell_count: int

    @property
    def tile_b(self) -> int:
        return self.tile_keys[0][0]

    @property
    def tile_c(self) -> int:
        return self.tile_keys[0][1]


@dataclass(frozen=True)
class ProductionLayoutValidation:
    valid: bool
    cell_count: int
    chunk_count: int
    smallest_chunk: int
    largest_chunk: int
    checked_chunks: int
    connected_chunks: bool
    issues: tuple[str, ...]


class _ChunkSequence(Sequence[ChunkDefinition]):
    def __init__(self, layout: "ProductionChunkLayout") -> None:
        self.layout = layout

    def __len__(self) -> int:
        return len(self.layout.records)

    @overload
    def __getitem__(self, index: int) -> ChunkDefinition: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ChunkDefinition, ...]: ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.layout.records[index]
        return ChunkDefinition(record.chunk_id, record.base_face, self.layout.chunk_cell_ids(index))


class _CellToChunkSequence(Sequence[int]):
    def __init__(self, layout: "ProductionChunkLayout", local: bool) -> None:
        self.layout = layout
        self.local = local

    def __len__(self) -> int:
        return self.layout.cell_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            values = range(*index.indices(len(self)))
            if self.local:
                return tuple(self.layout.chunk_for_cell(cell_id)[1] for cell_id in values)
            return tuple(self.layout.chunk_for_cell(cell_id)[0] for cell_id in values)
        result = self.layout.chunk_for_cell(index)
        return result[1] if self.local else result[0]


class ProductionChunkLayout:
    """Compact face-tile layout whose chunk members are generated on demand."""

    generator_version = PRODUCTION_GENERATOR_VERSION
    orientation = PRODUCTION_ORIENTATION
    layout_version = PRODUCTION_LAYOUT_VERSION
    layout_mode = "procedural_face_tiles"

    def __init__(
        self,
        topology: ProductionTopology,
        tile_side: int = DEFAULT_TILE_SIDE,
        records: tuple[ProductionChunkRecord, ...] | None = None,
        stable_hash: str | None = None,
    ) -> None:
        if tile_side < 2 or tile_side > 255:
            raise ChunkLayoutError("Production tile side must be between 2 and 255")
        self.topology = topology
        self.frequency = topology.frequency
        self.topology_hash = topology.stable_hash
        self.tile_side = int(tile_side)
        self.target_cells = self.tile_side * self.tile_side
        if records is None:
            records = self._build_records()
        self.records = records
        self._key_to_chunk = {
            (record.base_face, tile_b, tile_c): record.chunk_id
            for record in records
            for tile_b, tile_c in record.tile_keys
        }
        self.stable_hash = stable_hash or self._stable_hash()
        self.chunks = _ChunkSequence(self)
        self.cell_to_chunk = _CellToChunkSequence(self, local=False)
        self.cell_to_local_index = _CellToChunkSequence(self, local=True)

    @property
    def cell_count(self) -> int:
        return self.topology.cell_count

    @property
    def chunk_count(self) -> int:
        return len(self.records)

    @property
    def bin_count(self) -> int:
        return max(1, (self.frequency + self.tile_side - 1) // self.tile_side)

    def _tile_index(self, coordinate: int) -> int:
        return min(coordinate // self.tile_side, self.bin_count - 1)

    def _tile_bounds(self, tile_index: int) -> tuple[int, int]:
        if tile_index < 0 or tile_index >= self.bin_count:
            raise IndexError(tile_index)
        start = tile_index * self.tile_side
        end = self.frequency if tile_index == self.bin_count - 1 else min(
            self.frequency, (tile_index + 1) * self.tile_side - 1
        )
        return start, end

    @staticmethod
    def _owner_face_for_weights(
        topology: ProductionTopology,
        face_id: int,
        weight_a: int,
        weight_b: int,
        weight_c: int,
        cell_id: int,
    ) -> int:
        if weight_a > 0 and weight_b > 0 and weight_c > 0:
            return face_id
        return topology.owner_address(cell_id).face_id

    def _iter_tile_cells(self, face_id: int, tile_b: int, tile_c: int) -> Iterator[int]:
        b_start, b_end = self._tile_bounds(tile_b)
        c_start, c_end = self._tile_bounds(tile_c)
        for weight_b in range(b_start, b_end + 1):
            maximum_c = self.frequency - weight_b
            if maximum_c < c_start:
                continue
            for weight_c in range(c_start, min(c_end, maximum_c) + 1):
                weight_a = self.frequency - weight_b - weight_c
                cell_id = self.topology.point_id(face_id, weight_a, weight_b, weight_c)
                if self._owner_face_for_weights(
                    self.topology, face_id, weight_a, weight_b, weight_c, cell_id
                ) == face_id:
                    yield cell_id

    @staticmethod
    def _neighbor_tile_keys(tile_b: int, tile_c: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (tile_b + delta_b, tile_c + delta_c)
            for delta_b, delta_c in ((-1, 0), (0, -1), (0, 1), (1, 0))
        )

    def _build_records(self) -> tuple[ProductionChunkRecord, ...]:
        records: list[ProductionChunkRecord] = []
        minimum_cells = max(1, self.target_cells // 2)
        maximum_cells = self.target_cells + self.target_cells // 2
        for face_id in range(20):
            counts: dict[tuple[int, int], int] = {}
            for tile_b in range(self.bin_count):
                b_start, _ = self._tile_bounds(tile_b)
                for tile_c in range(self.bin_count):
                    c_start, _ = self._tile_bounds(tile_c)
                    if b_start + c_start > self.frequency:
                        continue
                    count = sum(1 for _ in self._iter_tile_cells(face_id, tile_b, tile_c))
                    if count:
                        counts[(tile_b, tile_c)] = count

            groups: dict[int, set[tuple[int, int]]] = {}
            group_counts: dict[int, int] = {}
            group_of_key: dict[tuple[int, int], int] = {}
            for group_id, key in enumerate(sorted(counts)):
                groups[group_id] = {key}
                group_counts[group_id] = counts[key]
                group_of_key[key] = group_id

            while True:
                small_ids = [
                    group_id for group_id, count in group_counts.items()
                    if count < minimum_cells and len(group_counts) > 1
                ]
                if not small_ids:
                    break
                small_id = min(small_ids, key=lambda item: (group_counts[item], min(groups[item]), item))
                neighbor_ids: set[int] = set()
                for tile_b, tile_c in groups[small_id]:
                    for neighbor_key in self._neighbor_tile_keys(tile_b, tile_c):
                        other_id = group_of_key.get(neighbor_key)
                        if other_id is not None and other_id != small_id:
                            neighbor_ids.add(other_id)
                if not neighbor_ids:
                    break
                small_count = group_counts[small_id]
                preferred = [
                    other_id for other_id in neighbor_ids
                    if small_count + group_counts[other_id] <= maximum_cells
                ]
                choices = preferred or list(neighbor_ids)
                target_id = min(
                    choices,
                    key=lambda item: (
                        small_count + group_counts[item] > maximum_cells,
                        group_counts[item],
                        min(groups[item]),
                        item,
                    ),
                )
                for key in groups[small_id]:
                    groups[target_id].add(key)
                    group_of_key[key] = target_id
                group_counts[target_id] += group_counts[small_id]
                del groups[small_id]
                del group_counts[small_id]

            ordered_groups = sorted(groups, key=lambda item: (min(groups[item]), item))
            for group_id in ordered_groups:
                keys = tuple(sorted(groups[group_id]))
                records.append(
                    ProductionChunkRecord(
                        chunk_id=len(records),
                        base_face=face_id,
                        tile_keys=keys,
                        cell_count=group_counts[group_id],
                    )
                )
        if sum(record.cell_count for record in records) != self.topology.cell_count:
            raise ChunkLayoutError("Production chunk records do not cover every CellId exactly once")
        return tuple(records)

    def _stable_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            struct.pack(
                "<IIIII",
                self.layout_version,
                self.frequency,
                self.tile_side,
                self.target_cells,
                self.chunk_count,
            )
        )
        digest.update(bytes.fromhex(self.topology_hash))
        digest.update(b"canonical_owner_min_face_balanced_barycentric_tiles_v2")
        for record in self.records:
            digest.update(struct.pack("<HII", record.base_face, record.cell_count, len(record.tile_keys)))
            for tile_b, tile_c in record.tile_keys:
                digest.update(PRODUCTION_TILE_KEY.pack(tile_b, tile_c))
        return digest.hexdigest()

    @lru_cache(maxsize=4096)
    def chunk_cell_ids(self, chunk_id: int) -> tuple[int, ...]:
        if chunk_id < 0 or chunk_id >= self.chunk_count:
            raise IndexError(f"ChunkId is outside layout: {chunk_id}")
        record = self.records[chunk_id]
        members = tuple(
            cell_id
            for tile_b, tile_c in record.tile_keys
            for cell_id in self._iter_tile_cells(record.base_face, tile_b, tile_c)
        )
        if len(members) != record.cell_count:
            raise ChunkLayoutError(
                f"Chunk {chunk_id} count mismatch: expected {record.cell_count}, received {len(members)}"
            )
        return members

    @lru_cache(maxsize=4096)
    def _chunk_local_map(self, chunk_id: int) -> dict[int, int]:
        return {cell_id: index for index, cell_id in enumerate(self.chunk_cell_ids(chunk_id))}

    def chunk_for_cell(self, cell_id: int) -> tuple[int, int]:
        cell_id = int(cell_id)
        if (
            cell_id >= self.topology.face_start
            and self.topology.interior_per_face > 0
        ):
            offset = cell_id - self.topology.face_start
            face_id, rank = divmod(offset, self.topology.interior_per_face)
            row_index = bisect.bisect_right(self.topology._row_starts, rank) - 1
            weight_b = row_index + 1
            weight_c = rank - self.topology._row_starts[row_index] + 1
        else:
            address = self.topology.owner_address(cell_id)
            face_id = address.face_id
            weight_b = address.weight_b
            weight_c = address.weight_c
        key = (
            face_id,
            self._tile_index(weight_b),
            self._tile_index(weight_c),
        )
        try:
            chunk_id = self._key_to_chunk[key]
        except KeyError as exc:
            raise ChunkLayoutError(f"No production chunk owns CellId {cell_id}") from exc
        try:
            local_index = self._chunk_local_map(chunk_id)[cell_id]
        except KeyError as exc:
            raise ChunkLayoutError(f"Chunk {chunk_id} does not contain CellId {cell_id}") from exc
        return chunk_id, local_index

    def validate(
        self,
        *,
        full_chunk_limit: int = 2048,
        sample_chunks: int = 512,
    ) -> ProductionLayoutValidation:
        issues: list[str] = []
        counts = [record.cell_count for record in self.records]
        if not counts:
            issues.append("Production layout has no chunks")
            return ProductionLayoutValidation(
                False, self.cell_count, 0, 0, 0, 0, False, tuple(issues)
            )
        if sum(counts) != self.cell_count:
            issues.append("Chunk record counts do not equal topology cell count")
        if any(record.chunk_id != index for index, record in enumerate(self.records)):
            issues.append("Chunk ids are not contiguous")

        if self.chunk_count <= full_chunk_limit:
            checked_ids = tuple(range(self.chunk_count))
        else:
            step = max(1, self.chunk_count // sample_chunks)
            checked_ids = tuple(dict.fromkeys((*range(0, self.chunk_count, step), self.chunk_count - 1)))

        connected = True
        for chunk_id in checked_ids:
            members = self.chunk_cell_ids(chunk_id)
            if len(members) != len(set(members)):
                issues.append(f"Chunk {chunk_id} contains duplicate CellIds")
                connected = False
                continue
            member_set = set(members)
            visited = {members[0]}
            queue = deque([members[0]])
            while queue:
                cell_id = queue.popleft()
                for neighbor in self.topology.cell_neighbors(cell_id):
                    if neighbor in member_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(visited) != len(members):
                connected = False
                issues.append(
                    f"Chunk {chunk_id} is disconnected: reached {len(visited)} of {len(members)} cells"
                )

        expected_hash = self._stable_hash()
        if expected_hash != self.stable_hash:
            issues.append("Production layout stable hash mismatch")
        return ProductionLayoutValidation(
            valid=not issues,
            cell_count=self.cell_count,
            chunk_count=self.chunk_count,
            smallest_chunk=min(counts),
            largest_chunk=max(counts),
            checked_chunks=len(checked_ids),
            connected_chunks=connected,
            issues=tuple(issues),
        )


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


def write_production_layout_cache(
    layout: ProductionChunkLayout,
    topology_root: str | Path,
) -> ChunkLayoutCacheInfo:
    directory = Path(topology_root) / f"ico_dual_f{layout.frequency}_v2"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "production_chunks.idx"
    manifest_path = directory / "production.json"

    payload = bytearray(
        PRODUCTION_LAYOUT_HEADER.pack(
            PRODUCTION_LAYOUT_MAGIC,
            PRODUCTION_LAYOUT_VERSION,
            layout.frequency,
            layout.tile_side,
            layout.cell_count,
            layout.chunk_count,
            PRODUCTION_CHUNK_RECORD.size,
            bytes.fromhex(layout.topology_hash),
            bytes.fromhex(layout.stable_hash),
        )
    )
    tile_keys: list[tuple[int, int]] = []
    for record in layout.records:
        key_offset = len(tile_keys)
        tile_keys.extend(record.tile_keys)
        payload.extend(
            PRODUCTION_CHUNK_RECORD.pack(
                record.base_face,
                0,
                key_offset,
                len(record.tile_keys),
                record.cell_count,
            )
        )
    for tile_b, tile_c in tile_keys:
        payload.extend(PRODUCTION_TILE_KEY.pack(tile_b, tile_c))
    raw = bytes(payload)
    crc32 = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    sha256 = hashlib.sha256(raw).hexdigest()
    _atomic_write_bytes(index_path, raw)
    _atomic_write_json(
        manifest_path,
        {
            "format": PRODUCTION_LAYOUT_FORMAT,
            "formatVersion": PRODUCTION_LAYOUT_VERSION,
            "frequency": layout.frequency,
            "generatorVersion": layout.generator_version,
            "orientation": layout.orientation,
            "cellIdScheme": layout.topology.cell_id_scheme,
            "topologyHash": layout.topology_hash,
            "layoutHash": layout.stable_hash,
            "tileSide": layout.tile_side,
            "targetCells": layout.target_cells,
            "counts": {
                "cells": layout.cell_count,
                "triangles": layout.topology.triangle_count,
                "pentagons": 12,
                "hexagons": layout.topology.hexagon_count,
                "chunks": layout.chunk_count,
                "tileKeys": sum(len(record.tile_keys) for record in layout.records),
            },
            "files": {
                "index": {
                    "path": index_path.name,
                    "bytes": len(raw),
                    "crc32": crc32,
                    "sha256": sha256,
                }
            },
            "storage": {
                "topology": "procedural_on_demand",
                "chunkMembers": "procedural_face_tile_enumeration",
                "materializedCellRecords": 0,
                "materializedTriangleRecords": 0,
            },
        },
    )
    return ChunkLayoutCacheInfo(directory, manifest_path, index_path, crc32, sha256)


def load_production_layout_cache(directory: str | Path) -> ProductionChunkLayout:
    cache_dir = Path(directory)
    try:
        manifest = json.loads((cache_dir / "production.json").read_text(encoding="utf-8"))
        payload = (cache_dir / str(manifest["files"]["index"]["path"])).read_bytes()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ChunkLayoutError(f"Cannot read production layout cache: {exc}") from exc
    if manifest.get("format") != PRODUCTION_LAYOUT_FORMAT:
        raise ChunkLayoutError("Unsupported production layout manifest")
    actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_crc != str(manifest["files"]["index"].get("crc32", "")).lower():
        raise ChunkLayoutError("Production layout CRC mismatch")
    if actual_sha != str(manifest["files"]["index"].get("sha256", "")).lower():
        raise ChunkLayoutError("Production layout SHA-256 mismatch")
    if len(payload) < PRODUCTION_LAYOUT_HEADER.size:
        raise ChunkLayoutError("Production layout header is truncated")
    (
        magic,
        version,
        frequency,
        tile_side,
        cell_count,
        chunk_count,
        record_size,
        topology_hash_raw,
        layout_hash_raw,
    ) = PRODUCTION_LAYOUT_HEADER.unpack_from(payload)
    if magic != PRODUCTION_LAYOUT_MAGIC or version != PRODUCTION_LAYOUT_VERSION:
        raise ChunkLayoutError("Unsupported production layout binary format")
    if record_size != PRODUCTION_CHUNK_RECORD.size:
        raise ChunkLayoutError("Unsupported production chunk record size")
    records_end = PRODUCTION_LAYOUT_HEADER.size + chunk_count * record_size
    if len(payload) < records_end:
        raise ChunkLayoutError("Production layout records are truncated")
    topology = ProductionTopology(frequency)
    if topology.cell_count != cell_count:
        raise ChunkLayoutError("Production layout cell count mismatch")
    if topology.stable_hash != topology_hash_raw.hex():
        raise ChunkLayoutError("Production topology hash mismatch")
    raw_records: list[tuple[int, int, int, int]] = []
    offset = PRODUCTION_LAYOUT_HEADER.size
    maximum_key_end = 0
    for chunk_id in range(chunk_count):
        face_id, flags, key_offset, key_count, count = PRODUCTION_CHUNK_RECORD.unpack_from(payload, offset)
        offset += record_size
        if flags != 0:
            raise ChunkLayoutError("Unsupported production chunk flags")
        raw_records.append((face_id, key_offset, key_count, count))
        maximum_key_end = max(maximum_key_end, key_offset + key_count)
    expected_size = records_end + maximum_key_end * PRODUCTION_TILE_KEY.size
    if len(payload) != expected_size:
        raise ChunkLayoutError(
            f"Production layout size mismatch: expected {expected_size}, received {len(payload)}"
        )
    all_keys = [
        PRODUCTION_TILE_KEY.unpack_from(payload, records_end + index * PRODUCTION_TILE_KEY.size)
        for index in range(maximum_key_end)
    ]
    records: list[ProductionChunkRecord] = []
    for chunk_id, (face_id, key_offset, key_count, count) in enumerate(raw_records):
        key_end = key_offset + key_count
        if key_end > len(all_keys) or key_count < 1:
            raise ChunkLayoutError(f"Production chunk {chunk_id} tile key range is invalid")
        records.append(
            ProductionChunkRecord(
                chunk_id, face_id, tuple(all_keys[key_offset:key_end]), count
            )
        )
    layout = ProductionChunkLayout(
        topology,
        tile_side=tile_side,
        records=tuple(records),
        stable_hash=layout_hash_raw.hex(),
    )
    if layout.stable_hash != str(manifest.get("layoutHash", "")):
        raise ChunkLayoutError("Production layout manifest hash mismatch")
    if layout._stable_hash() != layout.stable_hash:
        raise ChunkLayoutError("Production layout stable hash mismatch")
    return layout
