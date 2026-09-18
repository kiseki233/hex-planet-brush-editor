from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import time
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.gpu_batch import INSTANCE_STRUCT
from app.gpu_stream import VisibleGpuInstanceStream
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from tests.test_helpers import write_rgb_png


class VisibleGpuInstanceStreamTests(unittest.TestCase):
    def test_hardware_layer_limit_can_lower_an_idle_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stream = VisibleGpuInstanceStream(
                Path(temporary) / "brushes",
                maximum_texture_layers=4096,
            )
            stream.set_maximum_texture_layers(2048)
            self.assertEqual(stream.maximum_texture_layers, 2048)
            self.assertEqual(stream.residency.maximum_layers, 2048)

    def _fixture(self, temporary: str, frequency: int = 32):
        root = Path(temporary)
        brush_root = root / "brushes"
        write_rgb_png(brush_root / "地形" / "green.png", rgb=(20, 150, 60))
        record = BrushCatalog(brush_root).scan().active_records[0]
        topology = ProductionTopology(frequency)
        layout = ProductionChunkLayout(topology)
        store = SphereMapStore(root / "maps")
        session = store.create_blank("sphere", layout)
        first_chunk = next(
            chunk_id for chunk_id in range(layout.chunk_count)
            if any(cell_id >= 12 for cell_id in layout.chunks[chunk_id].cell_ids)
        )
        editable = next(cell_id for cell_id in layout.chunks[first_chunk].cell_ids if cell_id >= 12)
        session.set_cell(editable, store, record.uid, record.relative_path, 2)
        store.save(session)
        session.loaded_chunks.clear()
        return brush_root, topology, layout, store, session, {record.uid: record}, first_chunk, editable

    def test_stream_only_contains_requested_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, topology, layout, store, session, records, first_chunk, _ = self._fixture(temporary)
            stream = VisibleGpuInstanceStream(brush_root, lod_level=1)
            update = stream.update_visible_chunks(
                topology, layout, session, store, records, [first_chunk]
            )
            expected = sum(cell_id >= 12 for cell_id in layout.chunks[first_chunk].cell_ids)
            self.assertEqual(update.instance_count, expected)
            self.assertEqual(len(update.added), expected)
            self.assertEqual(session.loaded_chunks, {})
            self.assertLess(update.instance_count, topology.cell_count)

    def test_swap_remove_keeps_dense_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, topology, layout, store, session, records, first_chunk, _ = self._fixture(temporary)
            other_chunk = next(chunk_id for chunk_id in range(layout.chunk_count) if chunk_id != first_chunk)
            stream = VisibleGpuInstanceStream(brush_root)
            stream.update_visible_chunks(
                topology, layout, session, store, records, [first_chunk, other_chunk]
            )
            update = stream.update_visible_chunks(
                topology, layout, session, store, records, [other_chunk]
            )
            self.assertEqual(update.active_chunk_ids, (other_chunk,))
            self.assertEqual(set(stream.cell_to_slot.values()), set(range(stream.instance_count)))
            self.assertTrue(update.removed_cell_ids)
            self.assertEqual(len(stream.instance_bytes()), stream.instance_count * INSTANCE_STRUCT.size)

    def test_active_cell_patch_reuses_texture_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, topology, layout, store, session, records, first_chunk, editable = self._fixture(temporary)
            stream = VisibleGpuInstanceStream(brush_root)
            first = stream.update_visible_chunks(
                topology, layout, session, store, records, [first_chunk]
            )
            layer_count = len(first.texture_layers)
            session.set_cell(editable, store, next(iter(records)), next(iter(records.values())).relative_path, 5)
            patch = stream.patch_cell(
                topology, layout, session, store, records, editable
            )
            self.assertIsNotNone(patch)
            assert patch is not None
            self.assertEqual(patch.instance.rotation, 5)
            self.assertEqual(len(stream.layers), layer_count)

    def test_batch_patch_updates_multiple_visible_cells_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, topology, layout, store, session, records, first_chunk, editable = self._fixture(temporary)
            stream = VisibleGpuInstanceStream(brush_root)
            stream.update_visible_chunks(
                topology, layout, session, store, records, [first_chunk]
            )
            second = next(
                cell_id for cell_id in layout.chunks[first_chunk].cell_ids
                if cell_id >= 12 and cell_id != editable
            )
            record = next(iter(records.values()))
            session.set_cells(
                store,
                {
                    editable: (record.uid, record.relative_path, 4),
                    second: (record.uid, record.relative_path, 1),
                },
            )
            patches, released = stream.patch_cells(
                topology, layout, session, store, records, (editable, second)
            )
            self.assertEqual({item.cell_id for item in patches}, {editable, second})
            self.assertEqual(
                {item.instance.rotation for item in patches}, {1, 4}
            )
            self.assertEqual(released, ())

    def test_frequency_1004_stream_does_not_build_full_planet_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(1004)
            layout = ProductionChunkLayout(topology)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            stream = VisibleGpuInstanceStream(root / "brushes", lod_level=3)
            visible = [0, 1, 2]
            update = stream.update_visible_chunks(
                topology, layout, session, store, {}, visible
            )
            expected = sum(
                sum(cell_id >= 12 for cell_id in layout.chunks[chunk_id].cell_ids)
                for chunk_id in visible
            )
            self.assertEqual(update.instance_count, expected)
            self.assertLessEqual(update.instance_count, 3 * 384)
            self.assertLess(update.instance_count, topology.cell_count // 1000)
            self.assertEqual(session.loaded_chunks, {})

    def test_chunk_reads_use_multiple_worker_threads_but_merge_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                brush_root,
                topology,
                layout,
                store,
                session,
                records,
                _first_chunk,
                _editable,
            ) = self._fixture(temporary)
            chunk_ids = tuple(range(min(4, layout.chunk_count)))
            original_read = store.read_chunk_values
            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def delayed_read(current_session, chunk_id):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    time.sleep(0.03)
                    return original_read(current_session, chunk_id)
                finally:
                    with lock:
                        active -= 1

            store.read_chunk_values = delayed_read  # type: ignore[method-assign]
            stream = VisibleGpuInstanceStream(brush_root, lod_level=3)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                update = stream.update_visible_chunks(
                    topology,
                    layout,
                    session,
                    store,
                    records,
                    chunk_ids,
                    worker_pool=pool,
                )
            self.assertGreaterEqual(maximum_active, 2)
            self.assertEqual(update.active_chunk_ids, chunk_ids)
            self.assertEqual(
                [item.slot for item in update.added],
                list(range(update.instance_count)),
            )


if __name__ == "__main__":
    unittest.main()
