from __future__ import annotations

import json
import os
import re
import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MAP_FORMAT = "HEX_PLANET_MAP_LOCAL_PROTOTYPE"
MAP_FORMAT_VERSION = 1
DEFAULT_ROWS = 16
DEFAULT_COLUMNS = 16
MAX_LOCAL_BRUSH_ID = 4095


class MapFormatError(ValueError):
    pass


@dataclass
class BrushTableEntry:
    local_id: int
    brush_uid: str
    last_known_path: str


@dataclass
class LocalMap:
    name: str
    rows: int = DEFAULT_ROWS
    columns: int = DEFAULT_COLUMNS
    cells: list[int] = field(default_factory=lambda: [0] * (DEFAULT_ROWS * DEFAULT_COLUMNS))
    brush_entries: dict[int, BrushTableEntry] = field(default_factory=dict)
    dirty: bool = False

    def __post_init__(self) -> None:
        expected = self.rows * self.columns
        if len(self.cells) != expected:
            raise ValueError(f"Cell count mismatch: expected {expected}, received {len(self.cells)}")

    def get_state(self, index: int) -> tuple[int, int]:
        value = self.cells[index]
        local_id = value & 0x0FFF
        rotation = (value >> 12) & 0x0007
        return local_id, rotation

    def set_cell(self, index: int, brush_uid: str | None, last_known_path: str = "", rotation: int = 0) -> None:
        if rotation < 0 or rotation > 5:
            raise ValueError("Rotation must be between 0 and 5")
        if brush_uid is None:
            value = 0
        else:
            local_id = self._find_or_allocate_local_id(brush_uid, last_known_path)
            value = local_id | (rotation << 12)
        if self.cells[index] != value:
            self.cells[index] = value
            self.dirty = True

    def _find_or_allocate_local_id(self, brush_uid: str, last_known_path: str) -> int:
        for local_id, entry in self.brush_entries.items():
            if entry.brush_uid == brush_uid:
                if last_known_path and entry.last_known_path != last_known_path:
                    entry.last_known_path = last_known_path
                return local_id

        used = set(self.brush_entries)
        for local_id in range(1, MAX_LOCAL_BRUSH_ID + 1):
            if local_id not in used:
                self.brush_entries[local_id] = BrushTableEntry(local_id, brush_uid, last_known_path)
                return local_id
        raise MapFormatError("Map brush table is full")

    def brush_uid_for_cell(self, index: int) -> tuple[str | None, int]:
        local_id, rotation = self.get_state(index)
        if local_id == 0:
            return None, rotation
        entry = self.brush_entries.get(local_id)
        if entry is None:
            raise MapFormatError(f"Cell references missing local brush id {local_id}")
        return entry.brush_uid, rotation


class MapStore:
    def __init__(self, map_root: str | Path) -> None:
        self.map_root = Path(map_root)
        self.map_root.mkdir(parents=True, exist_ok=True)

    def create_blank(self, name: str) -> LocalMap:
        return LocalMap(name=self._validate_name(name))

    def list_maps(self) -> list[str]:
        names: list[str] = []
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
                names.append(item.name)
        return sorted(names, key=str.casefold)

    def save(self, local_map: LocalMap) -> Path:
        name = self._validate_name(local_map.name)
        map_dir = self.map_root / name
        data_dir = map_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        raw_cells = struct.pack(f"<{len(local_map.cells)}H", *local_map.cells)
        crc32 = zlib.crc32(raw_cells) & 0xFFFFFFFF
        brush_payload = {
            "version": 1,
            "entries": [
                {
                    "localId": entry.local_id,
                    "brushUid": entry.brush_uid,
                    "lastKnownPath": entry.last_known_path,
                }
                for entry in sorted(local_map.brush_entries.values(), key=lambda item: item.local_id)
            ],
        }
        map_payload = {
            "format": MAP_FORMAT,
            "formatVersion": MAP_FORMAT_VERSION,
            "name": name,
            "prototype": {
                "type": "planar_hex_grid",
                "rows": local_map.rows,
                "columns": local_map.columns,
                "purpose": "brush_and_editing_validation",
            },
            "cellEncoding": {
                "storage": "uint16_little_endian",
                "brushBits": "0-11",
                "rotationBits": "12-14",
                "reservedBit": 15,
            },
            "brushTable": "brush_table.json",
            "cellData": "data/cells.bin",
            "cellDataBytes": len(raw_cells),
            "crc32": f"{crc32:08x}",
        }

        self._atomic_write_bytes(data_dir / "cells.bin", raw_cells)
        self._atomic_write_json(map_dir / "brush_table.json", brush_payload)
        self._atomic_write_json(map_dir / "map.json", map_payload)
        local_map.name = name
        local_map.dirty = False
        return map_dir

    def load(self, name: str) -> LocalMap:
        valid_name = self._validate_name(name)
        map_dir = self.map_root / valid_name
        map_path = map_dir / "map.json"
        if not map_path.exists():
            raise MapFormatError(f"Map does not exist: {valid_name}")

        try:
            map_payload = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MapFormatError(f"Cannot read map.json: {exc}") from exc

        if map_payload.get("format") != MAP_FORMAT:
            raise MapFormatError("Unsupported map format")
        if map_payload.get("formatVersion") != MAP_FORMAT_VERSION:
            raise MapFormatError("Unsupported map format version")

        prototype = map_payload.get("prototype", {})
        rows = int(prototype.get("rows", 0))
        columns = int(prototype.get("columns", 0))
        if rows <= 0 or columns <= 0:
            raise MapFormatError("Invalid map dimensions")

        brush_table_path = map_dir / map_payload.get("brushTable", "brush_table.json")
        cell_data_path = map_dir / map_payload.get("cellData", "data/cells.bin")
        try:
            brush_payload = json.loads(brush_table_path.read_text(encoding="utf-8"))
            raw_cells = cell_data_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise MapFormatError(f"Cannot read map data: {exc}") from exc

        expected_bytes = rows * columns * 2
        if len(raw_cells) != expected_bytes:
            raise MapFormatError(f"Cell data length mismatch: expected {expected_bytes}, received {len(raw_cells)}")
        expected_crc = str(map_payload.get("crc32", "")).lower()
        actual_crc = f"{zlib.crc32(raw_cells) & 0xFFFFFFFF:08x}"
        if expected_crc != actual_crc:
            raise MapFormatError(f"Cell data CRC mismatch: expected {expected_crc}, received {actual_crc}")

        cells = list(struct.unpack(f"<{rows * columns}H", raw_cells))
        entries: dict[int, BrushTableEntry] = {}
        for item in brush_payload.get("entries", []):
            local_id = int(item["localId"])
            if local_id < 1 or local_id > MAX_LOCAL_BRUSH_ID:
                raise MapFormatError(f"Local brush id out of range: {local_id}")
            if local_id in entries:
                raise MapFormatError(f"Duplicate local brush id: {local_id}")
            entries[local_id] = BrushTableEntry(
                local_id=local_id,
                brush_uid=str(item["brushUid"]),
                last_known_path=str(item.get("lastKnownPath", "")),
            )

        for index, value in enumerate(cells):
            if value & 0x8000:
                raise MapFormatError(f"Reserved bit is set at cell {index}")
            local_id = value & 0x0FFF
            rotation = (value >> 12) & 0x0007
            if rotation in (6, 7):
                raise MapFormatError(f"Illegal rotation value {rotation} at cell {index}")
            if local_id != 0 and local_id not in entries:
                raise MapFormatError(f"Cell {index} references missing local brush id {local_id}")

        return LocalMap(
            name=valid_name,
            rows=rows,
            columns=columns,
            cells=cells,
            brush_entries=entries,
            dirty=False,
        )

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Map name cannot be empty")
        if normalized in {".", ".."}:
            raise ValueError("Invalid map name")
        if re.search(r"[\\/:*?\"<>|]", normalized):
            raise ValueError("Map name contains invalid path characters")
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
