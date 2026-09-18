from __future__ import annotations

import concurrent.futures
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from app.chunk_layout import build_chunk_layout
from app.brush_catalog import BrushRecord
from app.map_store import MapStore
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import (
    BLOCK_HEADER,
    ChunkDataError,
    SphereMapStore,
)
from app.topology import generate_dual_topology


class SphereMapStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(16)
        cls.layout = build_chunk_layout(cls.topology, target_cells=64)

    def test_production_blank_map_does_not_expand_all_chunk_cell_ids(self) -> None:
        topology = ProductionTopology(32)
        layout = ProductionChunkLayout(topology, tile_side=8)
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=32)
            with patch.object(
                ProductionChunkLayout,
                "chunk_cell_ids",
                side_effect=AssertionError("blank creation expanded procedural cell ids"),
            ):
                session = store.create_blank("planet", layout)
            self.assertEqual(len(session.index_records), layout.chunk_count)
            self.assertTrue(store.verify(session).valid)

    def test_blank_map_creates_index_and_packs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = SphereMapStore(root, chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            self.assertTrue((session.map_dir / "index.bin").exists())
            self.assertGreater(len(list((session.map_dir / "data").glob("pack_*.bin"))), 1)
            self.assertEqual(session.get_state(12, store), (0, 0))
            self.assertTrue(store.verify(session).valid)

    def test_incremental_save_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = SphereMapStore(root, chunks_per_pack=2)
            session = store.create_blank("sphere", self.layout)
            first_cell = self.layout.chunks[0].cell_ids[-1]
            if first_cell < 12:
                first_cell = next(cell for cell in self.layout.chunks[0].cell_ids if cell >= 12)
            final_cell = self.layout.chunks[-1].cell_ids[-1]
            before = {path.name: path.stat().st_size for path in (session.map_dir / "data").glob("pack_*.bin")}
            session.set_cell(first_cell, store, "uid-one", "城市/a.png", 1)
            session.set_cell(final_cell, store, "uid-two", "海洋/b.png", 5)
            dirty = set(session.dirty_chunks)
            store.save(session)
            self.assertFalse(session.dirty_chunks)
            after = {path.name: path.stat().st_size for path in (session.map_dir / "data").glob("pack_*.bin")}
            changed = {name for name in after if after[name] != before[name]}
            expected = {f"pack_{chunk_id // 2:04d}.bin" for chunk_id in dirty}
            self.assertEqual(changed, expected)

            reopened = store.open("sphere", self.layout)
            self.assertEqual(reopened.brush_uid_for_cell(first_cell, store), ("uid-one", 1))
            self.assertEqual(reopened.brush_uid_for_cell(final_cell, store), ("uid-two", 5))

    def test_incremental_save_fsyncs_once_per_dirty_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = SphereMapStore(root, chunks_per_pack=2)
            session = store.create_blank("sphere", self.layout)
            cells = [
                next(cell for cell in self.layout.chunks[chunk_id].cell_ids if cell >= 12)
                for chunk_id in (0, 1, 2)
            ]
            for cell in cells:
                session.set_cell(cell, store, "uid-one", "城市/a.png", 1)
            store.save(session)

            for cell in cells:
                session.set_cell(cell, store, "uid-one", "城市/a.png", 2)
            with patch("app.sphere_map_store.os.fsync", wraps=os.fsync) as fsync:
                store.save(session)

            # Chunks 0 and 1 share Pack 0, chunk 2 is in Pack 1, and the
            # atomically replaced index has one final durability barrier.
            self.assertEqual(fsync.call_count, 3)
            reopened = store.open("sphere", self.layout)
            for cell in cells:
                self.assertEqual(reopened.brush_uid_for_cell(cell, store), ("uid-one", 2))

    def test_incremental_save_reports_real_write_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=2)
            session = store.create_blank("sphere", self.layout)
            cell = next(
                cell_id
                for cell_id in self.layout.chunks[0].cell_ids
                if cell_id >= 12
            )
            session.set_cell(cell, store, "uid-one", "城市/a.png", 1)
            progress = []

            store.save(
                session,
                progress=lambda stage, completed, total: progress.append(
                    (stage, completed, total)
                ),
            )

            stages = [item[0] for item in progress]
            self.assertEqual(stages[0], "prepare")
            self.assertIn("pack", stages)
            self.assertIn("index", stages)
            self.assertEqual(stages[-1], "complete")

    def test_incremental_save_compresses_dirty_chunks_on_multiple_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=8)
            session = store.create_blank("sphere", self.layout)
            cells = [
                next(
                    cell
                    for cell in self.layout.chunks[chunk_id].cell_ids
                    if cell >= 12
                )
                for chunk_id in range(4)
            ]
            for cell in cells:
                session.set_cell(cell, store, "uid-parallel", "城市/p.png", 2)

            original_compress = store._compress_block_payload
            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def delayed_compress(raw):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    time.sleep(0.03)
                    return original_compress(raw)
                finally:
                    with lock:
                        active -= 1

            store._compress_block_payload = delayed_compress  # type: ignore[method-assign]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                store.save(session, worker_pool=pool)
            self.assertGreaterEqual(maximum_active, 2)
            reopened = store.open("sphere", self.layout)
            for cell in cells:
                self.assertEqual(
                    reopened.brush_uid_for_cell(cell, store),
                    ("uid-parallel", 2),
                )

    def test_random_paint_and_clear_bulk_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            cells = tuple(
                cell_id
                for chunk in self.layout.chunks[:3]
                for cell_id in chunk.cell_ids
                if cell_id >= 12
            )
            record = BrushRecord(
                uid="uid-random",
                relative_path="城市/random.png",
                content_hash="a" * 64,
                file_size=1,
                width=512,
                height=512,
                color_mode="RGB",
                category_path="城市",
                modified_time_ns=0,
                state="active",
                last_seen_utc=0,
                average_rgb=(10, 20, 30),
            )
            import random

            changed = session.paint_cells_random(
                store, cells, (record,), random.Random(7)
            )
            self.assertEqual(set(changed), set(cells))
            self.assertTrue(all(0 <= session.get_state(cell, store)[1] <= 5 for cell in cells))

            cleared = session.clear_cells(store, cells)
            self.assertEqual(set(cleared), set(cells))
            self.assertTrue(all(session.get_state(cell, store) == (0, 0) for cell in cells))

    def test_corrupt_one_chunk_is_reported_without_blocking_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = SphereMapStore(root, chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            target = session.index_records[1]
            pack_path = session.map_dir / "data" / f"pack_{target.pack_id:04d}.bin"
            with pack_path.open("r+b") as stream:
                stream.seek(target.offset + BLOCK_HEADER.size)
                value = stream.read(1)
                stream.seek(target.offset + BLOCK_HEADER.size)
                stream.write(bytes([value[0] ^ 0xFF]))

            reopened = store.open("sphere", self.layout)
            self.assertEqual(reopened.get_state(self.layout.chunks[0].cell_ids[-1], store), (0, 0))
            with self.assertRaises(ChunkDataError):
                store.load_chunk(reopened, 1)
            report = store.verify(reopened, [0, 1, 2])
            self.assertFalse(report.valid)
            self.assertEqual(report.failed_chunks, (1,))

    def test_illegal_rotation_and_reserved_bit_are_rejected_before_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            chunk_id, local_index = self.layout.chunk_for_cell(12)
            values = store.load_chunk(session, chunk_id)
            values[local_index] = 0x6000
            session.dirty_chunks.add(chunk_id)
            with self.assertRaises(ChunkDataError):
                store.save(session)
            values[local_index] = 0x8000
            with self.assertRaises(ChunkDataError):
                store.save(session)

    def test_viewport_chunk_sync_loads_releases_and_retains_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            loaded, unloaded, retained = store.sync_loaded_chunks(session, {0, 1})
            self.assertEqual(set(loaded), {0, 1})
            self.assertEqual(unloaded, ())
            self.assertEqual(retained, ())
            self.assertEqual(set(session.loaded_chunks), {0, 1})

            editable = next(cell_id for cell_id in self.layout.chunks[0].cell_ids if cell_id >= 12)
            session.set_cell(editable, store, "uid-dirty", "城市/dirty.png", 2)
            loaded, unloaded, retained = store.sync_loaded_chunks(session, {2})
            self.assertEqual(loaded, (2,))
            self.assertEqual(unloaded, (1,))
            self.assertEqual(retained, (0,))
            self.assertEqual(set(session.loaded_chunks), {0, 2})

            store.save(session)
            loaded, unloaded, retained = store.sync_loaded_chunks(session, {2})
            self.assertEqual(loaded, ())
            self.assertEqual(unloaded, (0,))
            self.assertEqual(retained, ())
            self.assertEqual(set(session.loaded_chunks), {2})

    def test_compatible_map_list_filters_topology_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            store = SphereMapStore(root)
            store.create_blank("sphere16", self.layout)
            other_topology = generate_dual_topology(8)
            other_layout = build_chunk_layout(other_topology, target_cells=64)
            store.create_blank("sphere8", other_layout)
            self.assertEqual(store.list_compatible_maps(self.layout), ["sphere16"])
            self.assertEqual(store.list_compatible_maps(other_layout), ["sphere8"])

    def test_local_map_list_does_not_include_sphere_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "maps"
            sphere_store = SphereMapStore(root)
            sphere_store.create_blank("sphere", self.layout)
            local_store = MapStore(root)
            local = local_store.create_blank("local")
            local_store.save(local)
            self.assertEqual(local_store.list_maps(), ["local"])
            self.assertEqual(sphere_store.list_maps(), ["sphere"])


if __name__ == "__main__":
    unittest.main()
