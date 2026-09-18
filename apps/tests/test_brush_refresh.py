from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.brush_lod import BrushLodCache
from app.gpu_batch import MISSING_LAYER_KEY, brush_texture_key
from app.production_layout import ProductionChunkLayout
from app.production_streaming import ProductionGpuStreamingController
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore
from tests.test_helpers import write_rgb_png


class BrushRefreshTest(unittest.TestCase):
    """Replacing a brush image and rescanning must re-texture what is on screen.

    A texture layer is keyed by uid *and* content hash, while instances hold a
    resolved layer index. Swapping the record dictionary alone left visible cells
    rendering the previous artwork until their chunk happened to be evicted.
    """

    def _fixture(self, temporary):
        root = Path(temporary)
        brush_root = root / "brushes"
        png = brush_root / "地形" / "a.png"
        write_rgb_png(png, rgb=(20, 150, 60))
        catalog = BrushCatalog(brush_root)
        records = {r.uid: r for r in catalog.scan().records}
        uid = next(iter(records))

        topology = ProductionTopology(32)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout)
        store = SphereMapStore(root / "maps")
        session = store.create_blank("planet", layout)
        chunk = next(
            c for c in range(layout.chunk_count)
            if any(x >= 12 for x in layout.chunks[c].cell_ids)
        )
        cells = [c for c in layout.chunks[chunk].cell_ids if c >= 12][:20]
        record = records[uid]
        session.set_cells(
            store, {c: (record.uid, record.relative_path, 0) for c in cells}
        )
        store.save(session)

        controller = ProductionGpuStreamingController(
            topology, layout, visibility, session, store, records, brush_root,
            lod_level=2, automatic_lod=False,
        )
        controller.stream.update_visible_chunks(
            topology, layout, session, store, records, [chunk]
        )
        return brush_root, png, catalog, controller, records, uid, chunk, session, store, topology, layout

    def _brush_layer_keys(self, controller):
        return {l.key for l in controller.stream.layers if l.kind == "brush"}

    def test_uid_survives_an_in_place_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, png, catalog, _, records, uid, *_ = self._fixture(temporary)
            before = records[uid].content_hash
            write_rgb_png(png, rgb=(200, 40, 40))
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertIn(uid, rescanned)
            self.assertNotEqual(rescanned[uid].content_hash, before)
            self.assertEqual(rescanned[uid].state, "active")

    def test_map_cells_still_resolve_after_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, png, catalog, _, _, uid, _, session, store, *_) = self._fixture(temporary)
            write_rgb_png(png, rgb=(200, 40, 40))
            rescanned = {r.uid: r for r in catalog.scan().records}
            local_id, _rotation = session.get_state(
                next(c for c in session.layout.chunks[0].cell_ids if c >= 12), store
            )
            for entry in session.brush_entries.values():
                self.assertIn(entry.brush_uid, rescanned)

    def test_refresh_resets_the_stream_and_reports_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, png, catalog, controller, records, uid, chunk, session, store,
             topology, layout) = self._fixture(temporary)
            old_key = brush_texture_key(records[uid], controller.stream.lod_level)
            self.assertIn(old_key, self._brush_layer_keys(controller))

            write_rgb_png(png, rgb=(200, 40, 40))
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertTrue(controller.update_records(rescanned))
            self.assertEqual(controller.stream.instance_count, 0)
            self.assertNotIn(old_key, self._brush_layer_keys(controller))

            controller.stream.update_visible_chunks(
                topology, layout, session, store, rescanned, [chunk]
            )
            new_key = brush_texture_key(rescanned[uid], controller.stream.lod_level)
            self.assertIn(new_key, self._brush_layer_keys(controller))
            self.assertNotIn(old_key, self._brush_layer_keys(controller))

    def test_unchanged_catalog_does_not_reset_the_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, _, catalog, controller, *_rest) = self._fixture(temporary)
            before = controller.stream.instance_count
            self.assertGreater(before, 0)
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertFalse(controller.update_records(rescanned))
            self.assertEqual(controller.stream.instance_count, before)

    def test_deleting_a_brush_marks_it_missing_and_resets(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, png, catalog, controller, _records, uid, chunk, session, store,
             topology, layout) = self._fixture(temporary)
            png.unlink()
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertEqual(rescanned[uid].state, "missing")
            self.assertTrue(controller.update_records(rescanned))

            controller.stream.update_visible_chunks(
                topology, layout, session, store, rescanned, [chunk]
            )
            # Cells fall back to the missing layer rather than losing their data.
            self.assertIn(MISSING_LAYER_KEY, {l.key for l in controller.stream.layers})
            for entry in session.brush_entries.values():
                self.assertEqual(entry.brush_uid, uid)

    def test_restoring_a_deleted_brush_recovers_the_artwork(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, png, catalog, controller, records, uid, chunk, session, store,
             topology, layout) = self._fixture(temporary)
            original_hash = records[uid].content_hash
            png.unlink()
            controller.update_records({r.uid: r for r in catalog.scan().records})
            write_rgb_png(png, rgb=(20, 150, 60))
            restored = {r.uid: r for r in catalog.scan().records}
            self.assertEqual(restored[uid].state, "active")
            self.assertEqual(restored[uid].content_hash, original_hash)
            controller.stream.update_visible_chunks(
                topology, layout, session, store, restored, [chunk]
            )
            self.assertIn(
                brush_texture_key(restored[uid], controller.stream.lod_level),
                self._brush_layer_keys(controller),
            )

    def test_forced_reset_reaches_the_next_frame_as_a_batch_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, png, catalog, controller, *_rest) = self._fixture(temporary)
            write_rgb_png(png, rgb=(200, 40, 40))
            controller.update_records({r.uid: r for r in catalog.scan().records})
            frame = controller.update_view(0.0, 0.0, 40.0, 1100, 760)
            self.assertIsNotNone(frame.reset_batch)


class BrushLodPruneTest(unittest.TestCase):
    def test_superseded_lod_directories_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            png = brush_root / "地形" / "a.png"
            write_rgb_png(png, rgb=(20, 150, 60))
            catalog = BrushCatalog(brush_root)
            record = catalog.scan().active_records[0]
            cache = BrushLodCache(brush_root)
            cache.ensure_all(record)
            safe_uid = record.uid.replace("/", "_")
            hash_dirs = lambda: sorted(
                p.name for p in (cache.cache_root / safe_uid).iterdir() if p.is_dir()
            )
            self.assertEqual(hash_dirs(), [record.content_hash])

            write_rgb_png(png, rgb=(200, 40, 40))
            replaced = catalog.scan().active_records[0]
            self.assertNotEqual(replaced.content_hash, record.content_hash)
            cache.ensure_all(replaced)
            self.assertEqual(hash_dirs(), [replaced.content_hash])

    def test_prune_keeps_the_current_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            write_rgb_png(brush_root / "地形" / "a.png", rgb=(20, 150, 60))
            record = BrushCatalog(brush_root).scan().active_records[0]
            cache = BrushLodCache(brush_root)
            cache.ensure_all(record)
            self.assertEqual(cache.prune_superseded(record), 0)
            for level in range(4):
                self.assertIsNotNone(cache.existing_path(record, level))


if __name__ == "__main__":
    unittest.main()
