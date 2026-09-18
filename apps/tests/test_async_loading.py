from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from app.async_loading import AsyncViewportChunkLoader
from app.chunk_layout import build_chunk_layout
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology


class AsyncLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = build_chunk_layout(generate_dual_topology(8), target_cells=32)

    def test_request_does_not_wait_for_disk_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            original = store.read_chunk_values

            def slow_read(current_session, chunk_id):
                time.sleep(0.12)
                return original(current_session, chunk_id)

            store.read_chunk_values = slow_read  # type: ignore[method-assign]
            loader = AsyncViewportChunkLoader(store, max_workers=1, max_inflight=2)
            try:
                started = time.perf_counter()
                update = loader.request(session, {0, 1})
                elapsed = time.perf_counter() - started
                self.assertLess(elapsed, 0.08)
                self.assertEqual(update.pending, 2)
                self.assertFalse(session.loaded_chunks)

                deadline = time.monotonic() + 2.0
                loaded: set[int] = set()
                while time.monotonic() < deadline and loaded != {0, 1}:
                    result = loader.poll(session)
                    loaded.update(result.loaded)
                    time.sleep(0.02)
                self.assertEqual(loaded, {0, 1})
                self.assertEqual(set(session.loaded_chunks), {0, 1})
            finally:
                loader.close()

    def test_unload_retains_dirty_and_releases_clean_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            store.load_chunk(session, 0)
            store.load_chunk(session, 1)
            editable = next(cell for cell in self.layout.chunks[0].cell_ids if cell >= 12)
            session.set_cell(editable, store, "uid", "地形/a.png", 3)
            loader = AsyncViewportChunkLoader(store)
            try:
                update = loader.request(session, {2})
                self.assertEqual(update.unloaded, (1,))
                self.assertEqual(update.retained_dirty, (0,))
                self.assertIn(0, session.loaded_chunks)
                self.assertNotIn(1, session.loaded_chunks)
                self.assertGreaterEqual(update.pending, 1)
            finally:
                loader.close()


    def test_failed_chunk_is_not_rescheduled_until_it_leaves_viewport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            calls = {"count": 0}

            def failing_read(current_session, chunk_id):
                calls["count"] += 1
                raise RuntimeError("synthetic read failure")

            store.read_chunk_values = failing_read  # type: ignore[method-assign]
            loader = AsyncViewportChunkLoader(store, max_workers=1, max_inflight=1)
            try:
                loader.request(session, {0})
                deadline = time.monotonic() + 1.0
                error_seen = False
                while time.monotonic() < deadline and not error_seen:
                    update = loader.poll(session)
                    error_seen = bool(update.errors)
                    time.sleep(0.01)
                self.assertTrue(error_seen)
                first_count = calls["count"]
                for _ in range(10):
                    loader.request(session, {0})
                    loader.poll(session)
                    time.sleep(0.01)
                self.assertEqual(calls["count"], first_count)
                loader.request(session, set())
                loader.request(session, {0})
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and calls["count"] == first_count:
                    loader.poll(session)
                    time.sleep(0.01)
                self.assertGreater(calls["count"], first_count)
            finally:
                loader.close()

    def test_read_chunk_values_does_not_mutate_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            values = store.read_chunk_values(session, 0)
            self.assertEqual(len(values), len(self.layout.chunks[0].cell_ids))
            self.assertNotIn(0, session.loaded_chunks)


if __name__ == "__main__":
    unittest.main()
