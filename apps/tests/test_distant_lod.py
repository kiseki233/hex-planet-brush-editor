from __future__ import annotations

import tempfile
import time
import threading
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.chunk_layout import build_chunk_layout
from app.distant_lod import (
    CHUNK_THUMBNAIL_SIZE,
    PLANET_OVERVIEW_HEIGHT,
    PLANET_OVERVIEW_WIDTH,
    AsyncDistantLodBuilder,
    DistantLodCache,
    _ChunkBuildInput,
    _PlanetBuildInput,
    chunk_convex_hulls,
)
from app.png_pixels import read_png_pixels
from app.sphere_map_store import SphereMapStore
from app.sphere_viewport import SphereViewport
from app.topology import generate_dual_topology
from tests.test_helpers import write_rgb_png


class DistantLodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(4)
        cls.layout = build_chunk_layout(cls.topology, target_cells=16)

    def _project(self, temporary: str):
        root = Path(temporary)
        brush_root = root / "brushes"
        map_root = root / "maps"
        source = brush_root / "海洋" / "red.png"
        write_rgb_png(source, rgb=(220, 30, 20))
        catalog = BrushCatalog(brush_root)
        record = catalog.scan().active_records[0]
        records = {record.uid: record}
        store = SphereMapStore(map_root, chunks_per_pack=4)
        session = store.create_blank("planet", self.layout)
        cell_id = next(cell for cell in self.layout.chunks[0].cell_ids if cell >= 12)
        session.set_cell(cell_id, store, record.uid, record.relative_path, 2)
        store.save(session)
        return brush_root, source, catalog, record, records, store, session, cell_id

    def test_chunk_thumbnail_cache_reuses_signature_and_invalidates_on_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, _source, _catalog, _record, records, store, session, cell_id = self._project(temporary)
            cache = DistantLodCache(brush_root)
            chunk_id, _ = self.layout.chunk_for_cell(cell_id)
            values = tuple(store.read_chunk_values(session, chunk_id))
            build_input = _ChunkBuildInput(
                session.name,
                session.map_dir,
                self.layout,
                self.topology,
                chunk_id,
                values,
                dict(session.brush_entries),
                records,
            )
            first = cache.ensure_chunk(build_input)
            second = cache.ensure_chunk(build_input)
            self.assertTrue(first.generated)
            self.assertFalse(second.generated)
            self.assertEqual(first.signature, second.signature)
            self.assertGreater(first.average_rgb[0], first.average_rgb[1])
            image = read_png_pixels(first.path)
            self.assertEqual((image.width, image.height), (CHUNK_THUMBNAIL_SIZE, CHUNK_THUMBNAIL_SIZE))
            self.assertEqual(image.channels, 4)

            session.set_cell(cell_id, store, None)
            changed_values = tuple(session.loaded_chunks[chunk_id])
            changed_input = _ChunkBuildInput(
                session.name,
                session.map_dir,
                self.layout,
                self.topology,
                chunk_id,
                changed_values,
                dict(session.brush_entries),
                records,
            )
            changed = cache.ensure_chunk(changed_input)
            self.assertTrue(changed.generated)
            self.assertNotEqual(first.signature, changed.signature)

    def test_representative_color_uses_catalog_value_without_source_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            source = brush_root / "海洋" / "color.png"
            write_rgb_png(source, rgb=(17, 83, 149))
            record = BrushCatalog(brush_root).scan().active_records[0]
            source.unlink()

            self.assertEqual(
                DistantLodCache(brush_root).representative_color(record),
                (17, 83, 149),
            )

    def test_same_uid_new_brush_content_invalidates_chunk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, source, catalog, record, records, store, session, cell_id = self._project(temporary)
            cache = DistantLodCache(brush_root)
            chunk_id, _ = self.layout.chunk_for_cell(cell_id)
            values = tuple(store.read_chunk_values(session, chunk_id))
            first = cache.ensure_chunk(
                _ChunkBuildInput(
                    session.name,
                    session.map_dir,
                    self.layout,
                    self.topology,
                    chunk_id,
                    values,
                    dict(session.brush_entries),
                    records,
                )
            )

            write_rgb_png(source, rgb=(20, 40, 230))
            updated = catalog.scan().active_records[0]
            self.assertEqual(record.uid, updated.uid)
            self.assertNotEqual(record.content_hash, updated.content_hash)
            second = cache.ensure_chunk(
                _ChunkBuildInput(
                    session.name,
                    session.map_dir,
                    self.layout,
                    self.topology,
                    chunk_id,
                    values,
                    dict(session.brush_entries),
                    {updated.uid: updated},
                )
            )
            self.assertNotEqual(first.signature, second.signature)
            self.assertGreater(second.average_rgb[2], second.average_rgb[0])

    def test_planet_overview_is_streamed_from_pack_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, _source, _catalog, _record, records, store, session, _cell_id = self._project(temporary)
            cache = DistantLodCache(brush_root)
            build_input = _PlanetBuildInput(
                session.name,
                session.map_dir,
                self.layout,
                self.topology,
                dict(session.brush_entries),
                records,
                tuple(session.index_records),
                session.chunks_per_pack,
            )
            first = cache.ensure_planet(build_input, store)
            second = cache.ensure_planet(build_input, store)
            self.assertTrue(first.generated)
            self.assertFalse(second.generated)
            self.assertEqual(first.signature, second.signature)
            image = read_png_pixels(first.path)
            self.assertEqual((image.width, image.height), (PLANET_OVERVIEW_WIDTH, PLANET_OVERVIEW_HEIGHT))
            self.assertEqual(image.channels, 3)
            self.assertIn(bytes((220, 30, 20)), image.pixels)

    def test_async_chunk_builder_is_bounded_and_completes_required_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, _source, _catalog, _record, records, store, session, _cell_id = self._project(temporary)
            builder = AsyncDistantLodBuilder(DistantLodCache(brush_root), store, max_workers=2, max_inflight=2)
            try:
                required = set(range(min(5, self.layout.chunk_count)))
                update = builder.request(session, self.topology, records, required)
                self.assertLessEqual(update.pending, 2)
                deadline = time.monotonic() + 5.0
                while set(builder.ready) != required and time.monotonic() < deadline:
                    builder.poll()
                    time.sleep(0.01)
                self.assertEqual(set(builder.ready), required)
                self.assertEqual(builder.pending_count(), 0)
            finally:
                builder.close()

    def test_invalidation_waits_for_running_writer_before_rebuild(self) -> None:
        class SlowFirstCache(DistantLodCache):
            def __init__(self, brush_root: Path) -> None:
                super().__init__(brush_root)
                self.started = threading.Event()
                self.release = threading.Event()
                self.calls = 0

            def ensure_chunk(self, build_input):  # type: ignore[override]
                self.calls += 1
                if self.calls == 1:
                    self.started.set()
                    self.release.wait(2.0)
                return super().ensure_chunk(build_input)

        with tempfile.TemporaryDirectory() as temporary:
            brush_root, _source, _catalog, record, records, store, session, cell_id = self._project(temporary)
            cache = SlowFirstCache(brush_root)
            builder = AsyncDistantLodBuilder(cache, store, max_workers=1, max_inflight=1)
            chunk_id, _ = self.layout.chunk_for_cell(cell_id)
            try:
                builder.request(session, self.topology, records, (chunk_id,))
                self.assertTrue(cache.started.wait(1.0))
                session.set_cell(cell_id, store, record.uid, record.relative_path, 5)
                builder.invalidate((chunk_id,))
                cache.release.set()
                deadline = time.monotonic() + 5.0
                while chunk_id not in builder.ready and time.monotonic() < deadline:
                    builder.poll()
                    time.sleep(0.01)
                self.assertIn(chunk_id, builder.ready)
                self.assertEqual(cache.calls, 2)
                current_values = tuple(session.loaded_chunks[chunk_id])
                expected = cache.chunk_signature(
                    self.layout, chunk_id, current_values, session.brush_entries, records
                )
                self.assertEqual(builder.ready[chunk_id].signature, expected)
            finally:
                cache.release.set()
                builder.close()

    def test_visible_cells_group_into_chunk_convex_hulls(self) -> None:
        projection = SphereViewport().project(self.topology, 800, 600, self.layout)
        hulls = chunk_convex_hulls(projection.cells, self.layout)
        self.assertTrue(hulls)
        self.assertTrue(set(hulls).issubset(set(projection.visible_chunk_ids)))
        for points in hulls.values():
            self.assertEqual(len(points) % 2, 0)
            self.assertGreaterEqual(len(points), 6)


if __name__ == "__main__":
    unittest.main()
