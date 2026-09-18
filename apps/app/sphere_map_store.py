from __future__ import annotations

import concurrent.futures
import json
import os
import re
import struct
import tempfile
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunk_layout import ChunkLayout
from .map_store import BrushTableEntry, MAX_LOCAL_BRUSH_ID
from .topology import GENERATOR_VERSION, ORIENTATION, TOPOLOGY_TYPE

MAP_FORMAT = "HEX_PLANET_MAP"
MAP_FORMAT_VERSION = 1
INDEX_MAGIC = b"HPINDEX1"
INDEX_VERSION = 1
INDEX_HEADER = struct.Struct("<8sIIIII32s32s")
INDEX_RECORD = struct.Struct("<IIQIIIIII")
PACK_MAGIC = b"HPPACK1\0"
PACK_VERSION = 1
PACK_HEADER = struct.Struct("<8sIII32s32s")
BLOCK_MAGIC = b"HPBLK01\0"
BLOCK_HEADER = struct.Struct("<8sIIIIII")
COMPRESSION_NONE = 0
COMPRESSION_ZLIB = 1
DEFAULT_CHUNKS_PER_PACK = 1024
FLAG_NONE = 0
COMPACTION_MARKER_NAME = ".pack_compaction.json"
COMPACTION_STAGE_NAME = ".pack_compaction_stage"
COMPACTION_BACKUP_DATA_NAME = ".pack_compaction_backup_data"
COMPACTION_BACKUP_INDEX_NAME = ".pack_compaction_backup_index.bin"
COMPACTION_TRANSACTION_VERSION = 1


class SphereMapError(ValueError):
    pass


class ChunkDataError(SphereMapError):
    def __init__(self, chunk_id: int, message: str) -> None:
        super().__init__(f"Chunk {chunk_id}: {message}")
        self.chunk_id = chunk_id


@dataclass(frozen=True)
class ChunkIndexRecord:
    chunk_id: int
    pack_id: int
    offset: int
    compressed_size: int
    raw_size: int
    compression_type: int
    crc32: int
    cell_count: int
    flags: int = FLAG_NONE


@dataclass
class SphereMapSession:
    name: str
    map_dir: Path
    layout: ChunkLayout
    brush_entries: dict[int, BrushTableEntry]
    index_records: list[ChunkIndexRecord]
    chunks_per_pack: int
    loaded_chunks: dict[int, list[int]] = field(default_factory=dict)
    dirty_chunks: set[int] = field(default_factory=set)
    brush_table_dirty: bool = False

    def get_state(self, cell_id: int, store: "SphereMapStore") -> tuple[int, int]:
        chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
        values = store.load_chunk(self, chunk_id)
        value = values[local_index]
        return value & 0x0FFF, (value >> 12) & 0x0007

    def brush_uid_for_cell(self, cell_id: int, store: "SphereMapStore") -> tuple[str | None, int]:
        local_id, rotation = self.get_state(cell_id, store)
        if local_id == 0:
            return None, rotation
        entry = self.brush_entries.get(local_id)
        if entry is None:
            raise SphereMapError(f"CellId {cell_id} references missing local brush id {local_id}")
        return entry.brush_uid, rotation

    def set_cell(
        self,
        cell_id: int,
        store: "SphereMapStore",
        brush_uid: str | None,
        last_known_path: str = "",
        rotation: int = 0,
    ) -> None:
        if cell_id < 12:
            raise SphereMapError("The 12 pentagon cells are reserved and cannot be painted")
        if rotation < 0 or rotation > 5:
            raise SphereMapError("Rotation must be between 0 and 5")
        chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
        values = store.load_chunk(self, chunk_id)
        if brush_uid is None:
            value = 0
        else:
            local_id = self._find_or_allocate_local_id(brush_uid, last_known_path)
            value = local_id | (rotation << 12)
        if values[local_index] != value:
            values[local_index] = value
            self.dirty_chunks.add(chunk_id)

    def set_cells(
        self,
        store: "SphereMapStore",
        assignments: dict[int, tuple[str, str, int] | None],
        *,
        skip_reserved: bool = True,
        journal=None,
    ) -> tuple[int, ...]:
        """Apply many complete cell replacements while loading each chunk once.

        ``journal`` is called with ``(chunk_id, values)`` after a chunk is loaded
        and before it is modified.  Undo capture rides on the grouping pass this
        method already performs; doing it separately would mean a second
        ``chunk_for_cell`` call for every affected cell, which on a 500-cell
        brush is 188,000 redundant lookups.
        """
        if not assignments:
            return ()

        value_by_uid: dict[tuple[str, str], int] = {}
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for raw_cell_id, assignment in assignments.items():
            cell_id = int(raw_cell_id)
            if cell_id < 12:
                if skip_reserved:
                    continue
                raise SphereMapError("The 12 pentagon cells are reserved and cannot be painted")
            chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
            if assignment is None:
                value = 0
            else:
                brush_uid, last_known_path, rotation = assignment
                rotation = int(rotation)
                if rotation < 0 or rotation > 5:
                    raise SphereMapError("Rotation must be between 0 and 5")
                cache_key = (brush_uid, last_known_path)
                local_id = value_by_uid.get(cache_key)
                if local_id is None:
                    local_id = self._find_or_allocate_local_id(brush_uid, last_known_path)
                    value_by_uid[cache_key] = local_id
                value = local_id | (rotation << 12)
            grouped.setdefault(chunk_id, []).append((local_index, value, cell_id))

        return self._apply_grouped_updates(store, grouped, journal)

    def paint_cells_random(
        self,
        store: "SphereMapStore",
        cell_ids,
        records,
        rng,
        *,
        journal=None,
    ) -> tuple[int, ...]:
        """Assign a random record and rotation while grouping cells in one pass."""
        records = tuple(records)
        if not records:
            raise SphereMapError("The selected brush group has no active images")
        local_by_record: dict[tuple[str, str], int] = {}
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        record_count = len(records)
        for raw_cell_id in cell_ids:
            cell_id = int(raw_cell_id)
            if cell_id < 12:
                continue
            record = records[rng.randrange(record_count)]
            cache_key = (record.uid, record.relative_path)
            local_id = local_by_record.get(cache_key)
            if local_id is None:
                local_id = self._find_or_allocate_local_id(*cache_key)
                local_by_record[cache_key] = local_id
            value = local_id | (rng.randrange(6) << 12)
            chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
            grouped.setdefault(chunk_id, []).append((local_index, value, cell_id))
        return self._apply_grouped_updates(store, grouped, journal)

    def clear_cells(
        self,
        store: "SphereMapStore",
        cell_ids,
        *,
        journal=None,
    ) -> tuple[int, ...]:
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for raw_cell_id in cell_ids:
            cell_id = int(raw_cell_id)
            if cell_id < 12:
                continue
            chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
            grouped.setdefault(chunk_id, []).append((local_index, 0, cell_id))
        return self._apply_grouped_updates(store, grouped, journal)

    def _apply_grouped_updates(
        self,
        store: "SphereMapStore",
        grouped: dict[int, list[tuple[int, int, int]]],
        journal,
    ) -> tuple[int, ...]:
        store.load_chunks(self, grouped)
        changed: list[int] = []
        for chunk_id, updates in grouped.items():
            values = store.load_chunk(self, chunk_id)
            if journal is not None:
                journal(chunk_id, values)
            chunk_changed = False
            for local_index, value, cell_id in updates:
                if values[local_index] == value:
                    continue
                values[local_index] = value
                chunk_changed = True
                changed.append(cell_id)
            if chunk_changed:
                self.dirty_chunks.add(chunk_id)
        return tuple(changed)

    def set_raw_values(
        self,
        store: "SphereMapStore",
        values_by_cell: dict[int, int],
        *,
        skip_reserved: bool = True,
        journal=None,
    ) -> tuple[int, ...]:
        """Write already-encoded uint16 cell states without touching the brush table.

        The undo brush restores raw snapshot values.  Those values reference local
        brush ids that this session allocated earlier, so no allocation is needed
        and none must happen: reallocating would hand out a different id for the
        same brush and break the cells that still reference the original.
        """
        if not values_by_cell:
            return ()

        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for raw_cell_id, raw_value in values_by_cell.items():
            cell_id = int(raw_cell_id)
            if cell_id < 12:
                if skip_reserved:
                    continue
                raise SphereMapError("The 12 pentagon cells are reserved and cannot be painted")
            value = int(raw_value)
            if value < 0 or value > 0xFFFF:
                raise SphereMapError(f"Cell state {value} is outside the uint16 range")
            local_id = value & 0x0FFF
            if local_id and local_id not in self.brush_entries:
                raise SphereMapError(
                    f"CellId {cell_id} would reference missing local brush id {local_id}"
                )
            chunk_id, local_index = self.layout.chunk_for_cell(cell_id)
            grouped.setdefault(chunk_id, []).append((local_index, value, cell_id))

        changed: list[int] = []
        for chunk_id, updates in grouped.items():
            values = store.load_chunk(self, chunk_id)
            if journal is not None:
                journal(chunk_id, values)
            chunk_changed = False
            for local_index, value, cell_id in updates:
                if values[local_index] == value:
                    continue
                values[local_index] = value
                chunk_changed = True
                changed.append(cell_id)
            if chunk_changed:
                self.dirty_chunks.add(chunk_id)
        return tuple(changed)

    def _find_or_allocate_local_id(self, brush_uid: str, last_known_path: str) -> int:
        for local_id, entry in self.brush_entries.items():
            if entry.brush_uid == brush_uid:
                if last_known_path and entry.last_known_path != last_known_path:
                    entry.last_known_path = last_known_path
                    self.brush_table_dirty = True
                return local_id
        for local_id in range(1, MAX_LOCAL_BRUSH_ID + 1):
            if local_id not in self.brush_entries:
                self.brush_entries[local_id] = BrushTableEntry(local_id, brush_uid, last_known_path)
                self.brush_table_dirty = True
                return local_id
        raise SphereMapError("Map brush table is full")


@dataclass(frozen=True)
class SphereMapVerification:
    valid: bool
    checked_chunks: int
    failed_chunks: tuple[int, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class PackStorageAnalysis:
    pack_count: int
    physical_bytes: int
    live_bytes: int
    reclaimable_bytes: int
    physical_blocks: int
    live_blocks: int
    orphan_blocks: int
    issues: tuple[str, ...]

    @property
    def reclaimable_ratio(self) -> float:
        if self.physical_bytes <= 0:
            return 0.0
        return self.reclaimable_bytes / self.physical_bytes


@dataclass(frozen=True)
class PackCompactionReport:
    before: PackStorageAnalysis
    after: PackStorageAnalysis
    bytes_reclaimed: int
    verified_chunks: int


@dataclass(frozen=True)
class PackRecoveryState:
    required: bool
    phase: str
    details: str


@dataclass(frozen=True)
class PackRecoveryReport:
    action: str
    verified_chunks: int


class SphereMapStore:
    def __init__(self, map_root: str | Path, chunks_per_pack: int = DEFAULT_CHUNKS_PER_PACK) -> None:
        if chunks_per_pack < 1:
            raise ValueError("chunks_per_pack must be positive")
        self.map_root = Path(map_root)
        self.map_root.mkdir(parents=True, exist_ok=True)
        self.chunks_per_pack = chunks_per_pack

    def list_maps(self) -> list[str]:
        result: list[str] = []
        for item in self.map_root.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue
            map_path = item / "map.json"
            if not map_path.exists():
                continue
            try:
                payload = json.loads(map_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("format") == MAP_FORMAT and payload.get("formatVersion") == MAP_FORMAT_VERSION:
                result.append(item.name)
        return sorted(result, key=str.casefold)

    def create_blank(self, name: str, layout: ChunkLayout) -> SphereMapSession:
        valid_name = self._validate_name(name)
        map_dir = self.map_root / valid_name
        if map_dir.exists() and any(map_dir.iterdir()):
            raise SphereMapError(f"Map directory already exists and is not empty: {valid_name}")
        data_dir = map_dir / "data"
        lod_dir = map_dir / "lod"
        data_dir.mkdir(parents=True, exist_ok=True)
        lod_dir.mkdir(parents=True, exist_ok=True)

        records: list[ChunkIndexRecord] = [None] * layout.chunk_count  # type: ignore[list-item]
        pack_count = (layout.chunk_count + self.chunks_per_pack - 1) // self.chunks_per_pack
        for pack_id in range(pack_count):
            first_chunk = pack_id * self.chunks_per_pack
            final_chunk = min(layout.chunk_count, first_chunk + self.chunks_per_pack)
            pack_path = data_dir / f"pack_{pack_id:04d}.bin"
            handle, temporary_name = tempfile.mkstemp(prefix=pack_path.name + ".", suffix=".tmp", dir=data_dir)
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(
                        PACK_HEADER.pack(
                            PACK_MAGIC,
                            PACK_VERSION,
                            pack_id,
                            self.chunks_per_pack,
                            bytes.fromhex(layout.topology_hash),
                            bytes.fromhex(layout.stable_hash),
                        )
                    )
                    blank_payload_cache: dict[int, tuple[bytes, int, int, int, int]] = {}
                    for chunk_id in range(first_chunk, final_chunk):
                        layout_records = getattr(layout, "records", None)
                        if layout_records is not None:
                            cell_count = int(layout_records[chunk_id].cell_count)
                        else:
                            cell_count = len(layout.chunks[chunk_id].cell_ids)
                        cached = blank_payload_cache.get(cell_count)
                        if cached is None:
                            raw = bytes(cell_count * 2)
                            compressed = zlib.compress(raw, level=6)
                            if len(compressed) < len(raw):
                                payload = compressed
                                compression_type = COMPRESSION_ZLIB
                            else:
                                payload = raw
                                compression_type = COMPRESSION_NONE
                            raw_size = len(raw)
                            compressed_size = len(payload)
                            crc32 = zlib.crc32(raw) & 0xFFFFFFFF
                            cached = (payload, compression_type, raw_size, compressed_size, crc32)
                            blank_payload_cache[cell_count] = cached
                        payload, compression_type, raw_size, compressed_size, crc32 = cached
                        offset = stream.tell()
                        stream.write(
                            BLOCK_HEADER.pack(
                                BLOCK_MAGIC,
                                chunk_id,
                                cell_count,
                                compression_type,
                                raw_size,
                                compressed_size,
                                crc32,
                            )
                        )
                        stream.write(payload)
                        records[chunk_id] = ChunkIndexRecord(
                            chunk_id=chunk_id,
                            pack_id=pack_id,
                            offset=offset,
                            compressed_size=compressed_size,
                            raw_size=raw_size,
                            compression_type=compression_type,
                            crc32=crc32,
                            cell_count=cell_count,
                            flags=FLAG_NONE,
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, pack_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        self._write_index(map_dir / "index.bin", layout, records, pack_count)
        self._write_brush_table(map_dir / "brush_table.json", {})
        self._write_map_manifest(map_dir / "map.json", valid_name, layout, pack_count)
        return SphereMapSession(
            name=valid_name,
            map_dir=map_dir,
            layout=layout,
            brush_entries={},
            index_records=records,
            chunks_per_pack=self.chunks_per_pack,
        )

    def open(
        self,
        name: str,
        layout: ChunkLayout,
        *,
        allow_recovery: bool = False,
    ) -> SphereMapSession:
        valid_name = self._validate_name(name)
        map_dir = self.map_root / valid_name
        if not allow_recovery:
            recovery = self.compaction_recovery_state(valid_name)
            if recovery.required:
                raise SphereMapError(
                    f"Pack compaction recovery is required ({recovery.phase}): {recovery.details}"
                )
        try:
            manifest = json.loads((map_dir / "map.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SphereMapError(f"Cannot read map.json: {exc}") from exc
        if manifest.get("format") != MAP_FORMAT or manifest.get("formatVersion") != MAP_FORMAT_VERSION:
            raise SphereMapError("Unsupported sphere map format")

        topology = manifest.get("topology", {})
        if int(topology.get("frequency", -1)) != layout.frequency:
            raise SphereMapError("Map topology frequency does not match the loaded chunk layout")
        if str(topology.get("stableHash", "")).lower() != layout.topology_hash:
            raise SphereMapError("Map topology hash does not match the loaded chunk layout")
        if str(manifest.get("chunkLayoutHash", "")).lower() != layout.stable_hash:
            raise SphereMapError("Map chunk layout hash does not match the loaded layout")

        chunks_per_pack = int(manifest.get("chunksPerPack", DEFAULT_CHUNKS_PER_PACK))
        records, pack_count = self._read_index(map_dir / str(manifest.get("chunkIndex", "index.bin")), layout)
        expected_pack_count = (layout.chunk_count + chunks_per_pack - 1) // chunks_per_pack
        manifest_pack_count = int(manifest.get("packCount", -1))
        if pack_count != expected_pack_count or manifest_pack_count != expected_pack_count:
            raise SphereMapError(
                f"Pack count mismatch: expected {expected_pack_count}, "
                f"manifest {manifest_pack_count}, index {pack_count}"
            )
        entries = self._read_brush_table(map_dir / str(manifest.get("brushTable", "brush_table.json")))
        return SphereMapSession(
            name=valid_name,
            map_dir=map_dir,
            layout=layout,
            brush_entries=entries,
            index_records=records,
            chunks_per_pack=chunks_per_pack,
        )

    def load_chunk(self, session: SphereMapSession, chunk_id: int) -> list[int]:
        if chunk_id in session.loaded_chunks:
            return session.loaded_chunks[chunk_id]
        values = self.read_chunk_values(session, chunk_id)
        session.loaded_chunks[chunk_id] = values
        return values

    def load_chunks(self, session: SphereMapSession, chunk_ids) -> None:
        """Populate several chunks while opening each Pack file only once."""
        pending_by_pack: dict[int, list[int]] = {}
        for raw_chunk_id in chunk_ids:
            chunk_id = int(raw_chunk_id)
            if chunk_id in session.loaded_chunks:
                continue
            if chunk_id < 0 or chunk_id >= len(session.index_records):
                raise ChunkDataError(chunk_id, "chunk id is outside index.bin")
            pack_id = session.index_records[chunk_id].pack_id
            pending_by_pack.setdefault(pack_id, []).append(chunk_id)

        for pack_id, pending in pending_by_pack.items():
            pack_path = session.map_dir / "data" / f"pack_{pack_id:04d}.bin"
            first_chunk = pending[0]
            try:
                with pack_path.open("rb") as stream:
                    self._validate_pack_stream(
                        session, pack_id, stream, first_chunk
                    )
                    for chunk_id in pending:
                        session.loaded_chunks[chunk_id] = self._read_chunk_from_stream(
                            session, chunk_id, stream
                        )
            except OSError as exc:
                raise ChunkDataError(
                    first_chunk, f"cannot read pack file: {exc}"
                ) from exc

    def read_chunk_values(self, session: SphereMapSession, chunk_id: int) -> list[int]:
        """Read and validate one chunk without mutating the session cache.

        This method is safe to run in a background worker as long as the session's
        immutable index/layout metadata is not replaced while the read is active.
        """
        if chunk_id < 0 or chunk_id >= len(session.index_records):
            raise ChunkDataError(chunk_id, "chunk id is outside index.bin")

        record = session.index_records[chunk_id]
        pack_path = session.map_dir / "data" / f"pack_{record.pack_id:04d}.bin"
        try:
            with pack_path.open("rb") as stream:
                self._validate_pack_stream(
                    session, record.pack_id, stream, chunk_id
                )
                return self._read_chunk_from_stream(session, chunk_id, stream)
        except OSError as exc:
            raise ChunkDataError(chunk_id, f"cannot read pack file: {exc}") from exc

    def read_chunk_values_many(
        self, session: SphereMapSession, chunk_ids
    ) -> dict[int, list[int]]:
        """Read many chunks, opening and validating each pack file only once.

        Summarising one far-view aggregate node reads hundreds of chunks that
        nearly all live in the same one or two pack files. Reading them one at a
        time reopened and re-validated the pack for every chunk, which measured
        as 632 file opens for a single node.
        """
        by_pack: dict[int, list[int]] = {}
        for raw_chunk_id in chunk_ids:
            chunk_id = int(raw_chunk_id)
            if chunk_id < 0 or chunk_id >= len(session.index_records):
                raise ChunkDataError(chunk_id, "chunk id is outside index.bin")
            by_pack.setdefault(session.index_records[chunk_id].pack_id, []).append(chunk_id)

        values_by_chunk: dict[int, list[int]] = {}
        for pack_id, ids in by_pack.items():
            pack_path = session.map_dir / "data" / f"pack_{pack_id:04d}.bin"
            try:
                with pack_path.open("rb") as stream:
                    self._validate_pack_stream(session, pack_id, stream, ids[0])
                    for chunk_id in ids:
                        values_by_chunk[chunk_id] = self._read_chunk_from_stream(
                            session, chunk_id, stream
                        )
            except OSError as exc:
                raise ChunkDataError(ids[0], f"cannot read pack file: {exc}") from exc
        return values_by_chunk

    @staticmethod
    def _validate_pack_stream(
        session: SphereMapSession,
        expected_pack_id: int,
        stream,
        chunk_id: int,
    ) -> None:
        stream.seek(0)
        header = stream.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise ChunkDataError(chunk_id, "pack header is truncated")
        magic, version, pack_id, chunks_per_pack, topology_hash, layout_hash = (
            PACK_HEADER.unpack(header)
        )
        if magic != PACK_MAGIC or version != PACK_VERSION:
            raise ChunkDataError(chunk_id, "pack format is unsupported")
        if pack_id != expected_pack_id or chunks_per_pack != session.chunks_per_pack:
            raise ChunkDataError(chunk_id, "pack metadata does not match index.bin")
        if topology_hash.hex() != session.layout.topology_hash:
            raise ChunkDataError(chunk_id, "pack topology hash mismatch")
        if layout_hash.hex() != session.layout.stable_hash:
            raise ChunkDataError(chunk_id, "pack layout hash mismatch")

    def _read_chunk_from_stream(
        self, session: SphereMapSession, chunk_id: int, stream
    ) -> list[int]:
        record = session.index_records[chunk_id]
        stream.seek(record.offset)
        block_header = stream.read(BLOCK_HEADER.size)
        if len(block_header) != BLOCK_HEADER.size:
            raise ChunkDataError(chunk_id, "block header is truncated")
        (
            block_magic,
            block_chunk_id,
            cell_count,
            compression_type,
            raw_size,
            compressed_size,
            crc32,
        ) = BLOCK_HEADER.unpack(block_header)
        if block_magic != BLOCK_MAGIC:
            raise ChunkDataError(chunk_id, "block magic is invalid")
        if (
            block_chunk_id != chunk_id
            or cell_count != record.cell_count
            or compression_type != record.compression_type
            or raw_size != record.raw_size
            or compressed_size != record.compressed_size
            or crc32 != record.crc32
        ):
            raise ChunkDataError(chunk_id, "block metadata does not match index.bin")
        compressed = stream.read(compressed_size)
        if len(compressed) != compressed_size:
            raise ChunkDataError(chunk_id, "block payload is truncated")

        if record.compression_type == COMPRESSION_NONE:
            raw = compressed
        elif record.compression_type == COMPRESSION_ZLIB:
            try:
                raw = zlib.decompress(compressed)
            except zlib.error as exc:
                raise ChunkDataError(chunk_id, f"zlib decompression failed: {exc}") from exc
        else:
            raise ChunkDataError(chunk_id, f"unsupported compression type {record.compression_type}")

        if len(raw) != record.raw_size:
            raise ChunkDataError(chunk_id, "raw chunk size mismatch")
        actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
        if actual_crc != record.crc32:
            raise ChunkDataError(
                chunk_id,
                f"CRC32 mismatch: expected {record.crc32:08x}, received {actual_crc:08x}",
            )
        if len(raw) != record.cell_count * 2:
            raise ChunkDataError(chunk_id, "cell byte count is invalid")

        values = list(struct.unpack(f"<{record.cell_count}H", raw))
        self._validate_cell_values(values, session.brush_entries, chunk_id)
        return values

    def sync_loaded_chunks(
        self,
        session: SphereMapSession,
        required_chunk_ids: set[int] | tuple[int, ...] | list[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        required = set(required_chunk_ids)
        invalid = sorted(
            chunk_id for chunk_id in required if chunk_id < 0 or chunk_id >= session.layout.chunk_count
        )
        if invalid:
            raise SphereMapError(f"Viewport requested invalid chunk ids: {invalid[:8]}")

        loaded_now: list[int] = []
        for chunk_id in sorted(required):
            if chunk_id not in session.loaded_chunks:
                self.load_chunk(session, chunk_id)
                loaded_now.append(chunk_id)

        retained_dirty: list[int] = []
        unloaded: list[int] = []
        for chunk_id in tuple(session.loaded_chunks):
            if chunk_id in required:
                continue
            if chunk_id in session.dirty_chunks:
                retained_dirty.append(chunk_id)
                continue
            session.loaded_chunks.pop(chunk_id, None)
            unloaded.append(chunk_id)

        return tuple(loaded_now), tuple(unloaded), tuple(sorted(retained_dirty))

    def list_compatible_maps(self, layout: ChunkLayout) -> list[str]:
        result: list[str] = []
        for name in self.list_maps():
            map_path = self.map_root / name / "map.json"
            try:
                payload = json.loads(map_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            topology = payload.get("topology", {})
            if int(topology.get("frequency", -1)) != layout.frequency:
                continue
            if str(topology.get("stableHash", "")).lower() != layout.topology_hash:
                continue
            if str(payload.get("chunkLayoutHash", "")).lower() != layout.stable_hash:
                continue
            result.append(name)
        return sorted(result, key=str.casefold)

    def save(
        self,
        session: SphereMapSession,
        progress: Callable[[str, int, int], None] | None = None,
        worker_pool: concurrent.futures.Executor | None = None,
    ) -> None:
        def report(stage: str, completed: int, total: int) -> None:
            if progress is None:
                return
            try:
                progress(stage, completed, total)
            except Exception:
                # Status reporting must never interrupt an atomic map save.
                pass

        if not session.dirty_chunks and not session.brush_table_dirty:
            report("complete", 0, 0)
            return
        report("prepare", 0, 0)
        if session.brush_table_dirty:
            report("brush_table", 0, 1)
            self._write_brush_table(session.map_dir / "brush_table.json", session.brush_entries)
            report("brush_table", 1, 1)

        updated_records = list(session.index_records)
        pending_by_pack: dict[int, list[tuple[int, bytes, int]]] = {}
        for chunk_id in sorted(session.dirty_chunks):
            values = session.loaded_chunks.get(chunk_id)
            if values is None:
                raise SphereMapError(f"Dirty chunk {chunk_id} is not loaded")
            self._validate_cell_values(values, session.brush_entries, chunk_id)
            raw = struct.pack(f"<{len(values)}H", *values)
            pack_id = chunk_id // session.chunks_per_pack
            pending_by_pack.setdefault(pack_id, []).append((chunk_id, raw, len(values)))

        # Appended blocks are not visible until index.bin is atomically replaced,
        # so all dirty chunks belonging to one Pack can share a single durability
        # barrier.  Flushing every tiny chunk separately made large brush saves
        # issue thousands of fsync calls and block for minutes on Windows.
        ordered_packs = sorted(pending_by_pack.items())
        pack_total = len(ordered_packs)
        local_pool = None
        if worker_pool is None and len(session.dirty_chunks) > 1:
            local_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=12,
                thread_name_prefix="hexplanet-save-compress",
            )
            worker_pool = local_pool
        try:
            for pack_index, (pack_id, pending) in enumerate(ordered_packs, start=1):
                report("pack", pack_index - 1, pack_total)
                if worker_pool is not None and len(pending) > 1:
                    compressed_items = worker_pool.map(
                        self._compress_block_payload,
                        (raw for _chunk_id, raw, _cell_count in pending),
                    )
                else:
                    compressed_items = map(
                        self._compress_block_payload,
                        (raw for _chunk_id, raw, _cell_count in pending),
                    )
                encoded = zip(pending, compressed_items)
                pack_path = session.map_dir / "data" / f"pack_{pack_id:04d}.bin"
                try:
                    with pack_path.open("ab") as stream:
                        for (
                            (chunk_id, raw, cell_count),
                            (payload, compression_type, crc32),
                        ) in encoded:
                            offset = stream.tell()
                            block, record = self._encode_precompressed_block(
                                chunk_id,
                                pack_id,
                                offset,
                                len(raw),
                                cell_count,
                                payload,
                                compression_type,
                                crc32,
                            )
                            stream.write(block)
                            updated_records[chunk_id] = record
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as exc:
                    raise SphereMapError(
                        f"Cannot append Pack {pack_id}: {exc}"
                    ) from exc
                report("pack", pack_index, pack_total)
        finally:
            if local_pool is not None:
                local_pool.shutdown(wait=True, cancel_futures=True)

        pack_count = (session.layout.chunk_count + session.chunks_per_pack - 1) // session.chunks_per_pack
        report("index", 0, 1)
        self._write_index(
            session.map_dir / "index.bin",
            session.layout,
            updated_records,
            pack_count,
        )
        session.index_records = updated_records
        session.dirty_chunks.clear()
        session.brush_table_dirty = False
        report("complete", pack_total, pack_total)

    def analyze_storage(self, session: SphereMapSession) -> PackStorageAnalysis:
        data_dir = session.map_dir / "data"
        physical_bytes = 0
        physical_blocks = 0
        issues: list[str] = []
        pack_files = sorted(data_dir.glob("pack_*.bin"))
        for path in pack_files:
            try:
                size = path.stat().st_size
                physical_bytes += size
                with path.open("rb") as stream:
                    header = stream.read(PACK_HEADER.size)
                    if len(header) != PACK_HEADER.size:
                        issues.append(f"{path.name}: pack header is truncated")
                        continue
                    magic, version, _pack_id, _chunks_per_pack, _topology_hash, _layout_hash = PACK_HEADER.unpack(header)
                    if magic != PACK_MAGIC or version != PACK_VERSION:
                        issues.append(f"{path.name}: unsupported pack header")
                        continue
                    while stream.tell() < size:
                        offset = stream.tell()
                        block_header = stream.read(BLOCK_HEADER.size)
                        if len(block_header) != BLOCK_HEADER.size:
                            issues.append(f"{path.name}: truncated block header at offset {offset}")
                            break
                        (
                            block_magic,
                            _chunk_id,
                            _cell_count,
                            _compression_type,
                            _raw_size,
                            compressed_size,
                            _crc32,
                        ) = BLOCK_HEADER.unpack(block_header)
                        if block_magic != BLOCK_MAGIC:
                            issues.append(f"{path.name}: invalid block magic at offset {offset}")
                            break
                        if compressed_size > size - stream.tell():
                            issues.append(f"{path.name}: truncated block payload at offset {offset}")
                            break
                        stream.seek(compressed_size, os.SEEK_CUR)
                        physical_blocks += 1
            except OSError as exc:
                issues.append(f"{path.name}: {exc}")

        expected_pack_count = (
            session.layout.chunk_count + session.chunks_per_pack - 1
        ) // session.chunks_per_pack
        live_bytes = expected_pack_count * PACK_HEADER.size
        live_bytes += sum(BLOCK_HEADER.size + record.compressed_size for record in session.index_records)
        reclaimable_bytes = max(0, physical_bytes - live_bytes)
        live_blocks = len(session.index_records)
        return PackStorageAnalysis(
            pack_count=len(pack_files),
            physical_bytes=physical_bytes,
            live_bytes=live_bytes,
            reclaimable_bytes=reclaimable_bytes,
            physical_blocks=physical_blocks,
            live_blocks=live_blocks,
            orphan_blocks=max(0, physical_blocks - live_blocks),
            issues=tuple(issues),
        )

    def compaction_recovery_state(self, name: str) -> PackRecoveryState:
        valid_name = self._validate_name(name)
        map_dir = self.map_root / valid_name
        marker_path = map_dir / COMPACTION_MARKER_NAME
        backup_data = map_dir / COMPACTION_BACKUP_DATA_NAME
        backup_index = map_dir / COMPACTION_BACKUP_INDEX_NAME
        stage_dir = map_dir / COMPACTION_STAGE_NAME
        if marker_path.exists():
            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
                phase = str(payload.get("phase", "unknown"))
            except (OSError, json.JSONDecodeError):
                phase = "unreadable"
            return PackRecoveryState(True, phase, "Compaction transaction requires recovery")
        leftovers = [path.name for path in (backup_data, backup_index, stage_dir) if path.exists()]
        if leftovers == [COMPACTION_STAGE_NAME] and (map_dir / "data").exists() and (map_dir / "index.bin").exists():
            return PackRecoveryState(
                True,
                "staging_without_marker",
                "An incomplete staging directory exists; official map files were not switched",
            )
        if leftovers:
            return PackRecoveryState(
                True,
                "orphaned_artifacts",
                "Compaction artifacts exist without a transaction marker: " + ", ".join(leftovers),
            )
        return PackRecoveryState(False, "none", "No interrupted compaction transaction")

    def compact(self, session: SphereMapSession) -> PackCompactionReport:
        if session.dirty_chunks or session.brush_table_dirty:
            raise SphereMapError("Save all dirty chunks and brush-table changes before Pack compaction")
        state = self.compaction_recovery_state(session.name)
        if state.required:
            raise SphereMapError(
                f"Cannot compact while recovery is required ({state.phase}): {state.details}"
            )

        before = self.analyze_storage(session)
        map_dir = session.map_dir
        stage_dir = map_dir / COMPACTION_STAGE_NAME
        stage_data = stage_dir / "data"
        stage_index = stage_dir / "index.bin"
        backup_data = map_dir / COMPACTION_BACKUP_DATA_NAME
        backup_index = map_dir / COMPACTION_BACKUP_INDEX_NAME
        marker_path = map_dir / COMPACTION_MARKER_NAME
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_data.mkdir(parents=True, exist_ok=True)

        original_records = list(session.index_records)
        new_records: list[ChunkIndexRecord] = [None] * session.layout.chunk_count  # type: ignore[list-item]
        pack_count = (session.layout.chunk_count + session.chunks_per_pack - 1) // session.chunks_per_pack
        try:
            for pack_id in range(pack_count):
                first_chunk = pack_id * session.chunks_per_pack
                final_chunk = min(session.layout.chunk_count, first_chunk + session.chunks_per_pack)
                pack_path = stage_data / f"pack_{pack_id:04d}.bin"
                with pack_path.open("wb") as stream:
                    stream.write(
                        PACK_HEADER.pack(
                            PACK_MAGIC,
                            PACK_VERSION,
                            pack_id,
                            session.chunks_per_pack,
                            bytes.fromhex(session.layout.topology_hash),
                            bytes.fromhex(session.layout.stable_hash),
                        )
                    )
                    for chunk_id in range(first_chunk, final_chunk):
                        values = self.read_chunk_values(session, chunk_id)
                        raw = struct.pack(f"<{len(values)}H", *values)
                        offset = stream.tell()
                        block, record = self._encode_block(
                            chunk_id, pack_id, offset, raw, len(values)
                        )
                        stream.write(block)
                        new_records[chunk_id] = record
                    stream.flush()
                    os.fsync(stream.fileno())

            self._write_index(stage_index, session.layout, new_records, pack_count)
            stage_session = SphereMapSession(
                name=session.name,
                map_dir=stage_dir,
                layout=session.layout,
                brush_entries=session.brush_entries,
                index_records=new_records,
                chunks_per_pack=session.chunks_per_pack,
            )
            stage_verification = self.verify(stage_session)
            if not stage_verification.valid:
                raise SphereMapError(
                    "Compacted staging data failed verification: "
                    + "; ".join(stage_verification.issues[:5])
                )

            self._write_compaction_marker(marker_path, "prepared")
            os.replace(map_dir / "data", backup_data)
            self._write_compaction_marker(marker_path, "old_data_moved")
            os.replace(map_dir / "index.bin", backup_index)
            self._write_compaction_marker(marker_path, "old_index_moved")
            os.replace(stage_data, map_dir / "data")
            self._write_compaction_marker(marker_path, "new_data_installed")
            os.replace(stage_index, map_dir / "index.bin")
            self._write_compaction_marker(marker_path, "new_index_installed")

            session.index_records = new_records
            verification = self.verify(session)
            if not verification.valid:
                session.index_records = original_records
                raise SphereMapError(
                    "Installed compacted data failed verification: "
                    + "; ".join(verification.issues[:5])
                )

            try:
                self._finalize_compaction_files(map_dir)
            except OSError:
                # The new data/index pair is already verified and authoritative.
                # A remaining recovery marker lets the next run finish cleanup.
                pass
            after = self.analyze_storage(session)
            return PackCompactionReport(
                before=before,
                after=after,
                bytes_reclaimed=max(0, before.physical_bytes - after.physical_bytes),
                verified_chunks=verification.checked_chunks,
            )
        except Exception:
            session.index_records = original_records
            try:
                self._rollback_compaction_files(map_dir, clear_marker=False)
                rollback_report = self.verify(session)
                if rollback_report.valid:
                    marker_path.unlink(missing_ok=True)
            except Exception:
                # Keep the marker and any remaining backup artifacts so recovery
                # can be retried instead of erasing evidence of an incomplete switch.
                pass
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
            raise

    def recover_compaction(self, name: str, layout: ChunkLayout) -> PackRecoveryReport:
        valid_name = self._validate_name(name)
        map_dir = self.map_root / valid_name
        marker_path = map_dir / COMPACTION_MARKER_NAME
        if not marker_path.exists():
            state = self.compaction_recovery_state(valid_name)
            if state.phase == "staging_without_marker":
                shutil.rmtree(map_dir / COMPACTION_STAGE_NAME)
                restored = self.open(valid_name, layout)
                report = self.verify(restored)
                if not report.valid:
                    raise SphereMapError(
                        "Official Pack data failed verification after discarding staging data: "
                        + "; ".join(report.issues[:5])
                    )
                return PackRecoveryReport("discarded_stage", report.checked_chunks)
            if state.required:
                raise SphereMapError(state.details)
            return PackRecoveryReport("none", 0)
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SphereMapError(f"Cannot read Pack compaction recovery marker: {exc}") from exc
        if int(payload.get("version", -1)) != COMPACTION_TRANSACTION_VERSION:
            raise SphereMapError("Unsupported Pack compaction recovery marker version")
        phase = str(payload.get("phase", "unknown"))

        if phase == "new_index_installed":
            try:
                candidate = self.open(valid_name, layout, allow_recovery=True)
                report = self.verify(candidate)
            except SphereMapError:
                report = SphereMapVerification(False, 0, (), ("new compacted files cannot open",))
            if report.valid:
                self._finalize_compaction_files(map_dir)
                return PackRecoveryReport("finalized_new", report.checked_chunks)

        self._rollback_compaction_files(map_dir, clear_marker=False)
        restored = self.open(valid_name, layout, allow_recovery=True)
        report = self.verify(restored)
        if not report.valid:
            raise SphereMapError(
                "Rolled-back Pack data failed verification: " + "; ".join(report.issues[:5])
            )
        marker_path.unlink(missing_ok=True)
        return PackRecoveryReport("rolled_back", report.checked_chunks)

    def _write_compaction_marker(self, path: Path, phase: str) -> None:
        self._atomic_write_json(
            path,
            {
                "version": COMPACTION_TRANSACTION_VERSION,
                "phase": phase,
                "stageDirectory": COMPACTION_STAGE_NAME,
                "backupDataDirectory": COMPACTION_BACKUP_DATA_NAME,
                "backupIndex": COMPACTION_BACKUP_INDEX_NAME,
            },
        )

    @staticmethod
    def _finalize_compaction_files(map_dir: Path) -> None:
        backup_data = map_dir / COMPACTION_BACKUP_DATA_NAME
        stage_dir = map_dir / COMPACTION_STAGE_NAME
        if backup_data.exists():
            shutil.rmtree(backup_data)
        (map_dir / COMPACTION_BACKUP_INDEX_NAME).unlink(missing_ok=True)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        (map_dir / COMPACTION_MARKER_NAME).unlink(missing_ok=True)

    @staticmethod
    def _rollback_compaction_files(map_dir: Path, *, clear_marker: bool = True) -> None:
        data_dir = map_dir / "data"
        index_path = map_dir / "index.bin"
        backup_data = map_dir / COMPACTION_BACKUP_DATA_NAME
        backup_index = map_dir / COMPACTION_BACKUP_INDEX_NAME
        if backup_data.exists():
            if data_dir.exists():
                shutil.rmtree(data_dir)
            os.replace(backup_data, data_dir)
        if backup_index.exists():
            index_path.unlink(missing_ok=True)
            os.replace(backup_index, index_path)
        stage_dir = map_dir / COMPACTION_STAGE_NAME
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if clear_marker:
            (map_dir / COMPACTION_MARKER_NAME).unlink(missing_ok=True)

    def verify(self, session: SphereMapSession, chunk_ids: list[int] | None = None) -> SphereMapVerification:
        targets = list(range(session.layout.chunk_count)) if chunk_ids is None else list(chunk_ids)
        failed: list[int] = []
        issues: list[str] = []
        for chunk_id in targets:
            was_loaded = chunk_id in session.loaded_chunks
            if was_loaded:
                session.loaded_chunks.pop(chunk_id, None)
            try:
                self.load_chunk(session, chunk_id)
            except SphereMapError as exc:
                failed.append(chunk_id)
                issues.append(str(exc))
            finally:
                if not was_loaded:
                    session.loaded_chunks.pop(chunk_id, None)
        return SphereMapVerification(
            valid=not failed,
            checked_chunks=len(targets),
            failed_chunks=tuple(failed),
            issues=tuple(issues),
        )

    @staticmethod
    def _validate_cell_values(
        values: list[int], brush_entries: dict[int, BrushTableEntry], chunk_id: int
    ) -> None:
        for local_index, value in enumerate(values):
            if value & 0x8000:
                raise ChunkDataError(chunk_id, f"reserved bit is set at local cell {local_index}")
            local_id = value & 0x0FFF
            rotation = (value >> 12) & 0x0007
            if rotation in (6, 7):
                raise ChunkDataError(chunk_id, f"illegal rotation {rotation} at local cell {local_index}")
            if local_id != 0 and local_id not in brush_entries:
                raise ChunkDataError(
                    chunk_id,
                    f"local cell {local_index} references missing brush id {local_id}",
                )

    @staticmethod
    def _encode_block(
        chunk_id: int,
        pack_id: int,
        offset: int,
        raw: bytes,
        cell_count: int,
    ) -> tuple[bytes, ChunkIndexRecord]:
        payload, compression_type, crc32 = SphereMapStore._compress_block_payload(
            raw
        )
        return SphereMapStore._encode_precompressed_block(
            chunk_id,
            pack_id,
            offset,
            len(raw),
            cell_count,
            payload,
            compression_type,
            crc32,
        )

    @staticmethod
    def _compress_block_payload(raw: bytes) -> tuple[bytes, int, int]:
        compressed = zlib.compress(raw, level=6)
        if len(compressed) < len(raw):
            payload = compressed
            compression_type = COMPRESSION_ZLIB
        else:
            payload = raw
            compression_type = COMPRESSION_NONE
        crc32 = zlib.crc32(raw) & 0xFFFFFFFF
        return payload, compression_type, crc32

    @staticmethod
    def _encode_precompressed_block(
        chunk_id: int,
        pack_id: int,
        offset: int,
        raw_size: int,
        cell_count: int,
        payload: bytes,
        compression_type: int,
        crc32: int,
    ) -> tuple[bytes, ChunkIndexRecord]:
        header = BLOCK_HEADER.pack(
            BLOCK_MAGIC,
            chunk_id,
            cell_count,
            compression_type,
            raw_size,
            len(payload),
            crc32,
        )
        record = ChunkIndexRecord(
            chunk_id=chunk_id,
            pack_id=pack_id,
            offset=offset,
            compressed_size=len(payload),
            raw_size=raw_size,
            compression_type=compression_type,
            crc32=crc32,
            cell_count=cell_count,
            flags=FLAG_NONE,
        )
        return header + payload, record

    def _write_index(
        self,
        path: Path,
        layout: ChunkLayout,
        records: list[ChunkIndexRecord],
        pack_count: int,
    ) -> None:
        if len(records) != layout.chunk_count:
            raise SphereMapError("Index record count does not match chunk layout")
        payload = bytearray(
            INDEX_HEADER.pack(
                INDEX_MAGIC,
                INDEX_VERSION,
                layout.cell_count,
                layout.chunk_count,
                pack_count,
                INDEX_RECORD.size,
                bytes.fromhex(layout.topology_hash),
                bytes.fromhex(layout.stable_hash),
            )
        )
        for expected_id, record in enumerate(records):
            if record.chunk_id != expected_id:
                raise SphereMapError(
                    f"Index record order mismatch: expected {expected_id}, received {record.chunk_id}"
                )
            payload.extend(
                INDEX_RECORD.pack(
                    record.chunk_id,
                    record.pack_id,
                    record.offset,
                    record.compressed_size,
                    record.raw_size,
                    record.compression_type,
                    record.crc32,
                    record.cell_count,
                    record.flags,
                )
            )
        self._atomic_write_bytes(path, bytes(payload))

    def _read_index(self, path: Path, layout: ChunkLayout) -> tuple[list[ChunkIndexRecord], int]:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SphereMapError(f"Cannot read index.bin: {exc}") from exc
        if len(payload) < INDEX_HEADER.size:
            raise SphereMapError("index.bin header is truncated")
        (
            magic,
            version,
            cell_count,
            chunk_count,
            pack_count,
            record_size,
            topology_hash,
            layout_hash,
        ) = INDEX_HEADER.unpack_from(payload)
        if magic != INDEX_MAGIC or version != INDEX_VERSION:
            raise SphereMapError("Unsupported index.bin format")
        if record_size != INDEX_RECORD.size:
            raise SphereMapError("Unsupported index record size")
        if cell_count != layout.cell_count or chunk_count != layout.chunk_count:
            raise SphereMapError("index.bin counts do not match the chunk layout")
        if topology_hash.hex() != layout.topology_hash or layout_hash.hex() != layout.stable_hash:
            raise SphereMapError("index.bin topology identity mismatch")
        expected_size = INDEX_HEADER.size + chunk_count * INDEX_RECORD.size
        if len(payload) != expected_size:
            raise SphereMapError(
                f"index.bin size mismatch: expected {expected_size}, received {len(payload)}"
            )

        records: list[ChunkIndexRecord] = []
        offset = INDEX_HEADER.size
        for expected_chunk_id in range(chunk_count):
            fields = INDEX_RECORD.unpack_from(payload, offset)
            offset += INDEX_RECORD.size
            record = ChunkIndexRecord(*fields)
            if record.chunk_id != expected_chunk_id:
                raise SphereMapError(
                    f"index.bin chunk order mismatch at {expected_chunk_id}: {record.chunk_id}"
                )
            expected_cells = len(layout.chunks[record.chunk_id].cell_ids)
            if record.cell_count != expected_cells or record.raw_size != expected_cells * 2:
                raise SphereMapError(f"index.bin cell count mismatch for chunk {record.chunk_id}")
            if record.pack_id < 0 or record.pack_id >= pack_count:
                raise SphereMapError(f"index.bin pack id out of range for chunk {record.chunk_id}")
            if record.offset < PACK_HEADER.size:
                raise SphereMapError(f"index.bin block offset is invalid for chunk {record.chunk_id}")
            if record.compressed_size < 0:
                raise SphereMapError(f"index.bin compressed size is invalid for chunk {record.chunk_id}")
            if record.compression_type not in (COMPRESSION_NONE, COMPRESSION_ZLIB):
                raise SphereMapError(
                    f"index.bin has unsupported compression for chunk {record.chunk_id}"
                )
            if record.flags != FLAG_NONE:
                raise SphereMapError(f"index.bin has unsupported flags for chunk {record.chunk_id}")
            records.append(record)
        return records, pack_count

    @staticmethod
    def _read_brush_table(path: Path) -> dict[int, BrushTableEntry]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SphereMapError(f"Cannot read brush_table.json: {exc}") from exc
        if payload.get("version") != 1:
            raise SphereMapError("Unsupported brush table version")
        entries: dict[int, BrushTableEntry] = {}
        for item in payload.get("entries", []):
            local_id = int(item["localId"])
            if local_id < 1 or local_id > MAX_LOCAL_BRUSH_ID:
                raise SphereMapError(f"Local brush id out of range: {local_id}")
            if local_id in entries:
                raise SphereMapError(f"Duplicate local brush id: {local_id}")
            entries[local_id] = BrushTableEntry(
                local_id=local_id,
                brush_uid=str(item["brushUid"]),
                last_known_path=str(item.get("lastKnownPath", "")),
            )
        return entries

    def _write_brush_table(self, path: Path, entries: dict[int, BrushTableEntry]) -> None:
        self._atomic_write_json(
            path,
            {
                "version": 1,
                "entries": [
                    {
                        "localId": entry.local_id,
                        "brushUid": entry.brush_uid,
                        "lastKnownPath": entry.last_known_path,
                    }
                    for entry in sorted(entries.values(), key=lambda item: item.local_id)
                ],
            },
        )

    def _write_map_manifest(
        self, path: Path, name: str, layout: ChunkLayout, pack_count: int
    ) -> None:
        self._atomic_write_json(
            path,
            {
                "format": MAP_FORMAT,
                "formatVersion": MAP_FORMAT_VERSION,
                "name": name,
                "planetDiameterKm": 5000.0,
                "cellFlatToFlatKm": 3.0,
                "topology": {
                    "type": getattr(layout, "topology_type", TOPOLOGY_TYPE),
                    "frequency": layout.frequency,
                    "generatorVersion": getattr(layout, "generator_version", GENERATOR_VERSION),
                    "orientation": getattr(layout, "orientation", ORIENTATION),
                    "hiddenPentagons": True,
                    "stableHash": layout.topology_hash,
                },
                "chunkTargetCells": layout.target_cells,
                "chunkLayoutHash": layout.stable_hash,
                "chunksPerPack": self.chunks_per_pack,
                "packCount": pack_count,
                "brushTable": "brush_table.json",
                "chunkIndex": "index.bin",
                "dataDirectory": "data",
                "lodDirectory": "lod",
                "cellEncoding": {
                    "storage": "uint16_little_endian",
                    "brushBits": "0-11",
                    "rotationBits": "12-14",
                    "reservedBit": 15,
                },
            },
        )

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise SphereMapError("Map name cannot be empty")
        if normalized in {".", ".."} or re.search(r"[\\/:*?\"<>|]", normalized):
            raise SphereMapError("Map name contains invalid path characters")
        return normalized

    @staticmethod
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

    @staticmethod
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
