from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.map_store import MapFormatError, MapStore


class MapStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MapStore(Path(temporary) / "maps")
            local_map = store.create_blank("测试地图")
            local_map.set_cell(0, "uid-one", "城市/city.png", 3)
            local_map.set_cell(255, "uid-two", "海洋/ocean.png", 5)
            store.save(local_map)
            loaded = store.load("测试地图")
            self.assertEqual(loaded.brush_uid_for_cell(0), ("uid-one", 3))
            self.assertEqual(loaded.brush_uid_for_cell(255), ("uid-two", 5))
            self.assertFalse(loaded.dirty)

    def test_same_uid_reuses_local_id(self) -> None:
        local_map = MapStore(Path(tempfile.gettempdir()) / "unused-map-root").create_blank("map")
        local_map.set_cell(0, "uid-one", "a.png", 0)
        local_map.set_cell(1, "uid-one", "a.png", 2)
        first_id, _ = local_map.get_state(0)
        second_id, _ = local_map.get_state(1)
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(local_map.brush_entries), 1)

    def test_crc_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = MapStore(root)
            local_map = store.create_blank("map")
            store.save(local_map)
            data_path = root / "map" / "data" / "cells.bin"
            payload = bytearray(data_path.read_bytes())
            payload[0] ^= 0xFF
            data_path.write_bytes(payload)
            with self.assertRaises(MapFormatError):
                store.load("map")

    def test_reserved_bit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = MapStore(root)
            local_map = store.create_blank("map")
            store.save(local_map)
            data_path = root / "map" / "data" / "cells.bin"
            payload = bytearray(data_path.read_bytes())
            payload[1] = 0x80
            data_path.write_bytes(payload)
            map_path = root / "map" / "map.json"
            map_payload = json.loads(map_path.read_text(encoding="utf-8"))
            import zlib

            map_payload["crc32"] = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
            map_path.write_text(json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(MapFormatError):
                store.load("map")

    def test_illegal_rotation_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = MapStore(root)
            local_map = store.create_blank("map")
            store.save(local_map)
            data_path = root / "map" / "data" / "cells.bin"
            payload = bytearray(data_path.read_bytes())
            payload[0] = 0
            payload[1] = 0x60
            data_path.write_bytes(payload)
            map_path = root / "map" / "map.json"
            map_payload = json.loads(map_path.read_text(encoding="utf-8"))
            import zlib

            map_payload["crc32"] = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
            map_path.write_text(json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(MapFormatError):
                store.load("map")


if __name__ == "__main__":
    unittest.main()
