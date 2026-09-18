from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.distant_lod import EMPTY_RGB
from app.production_layout import ProductionChunkLayout
from app.production_surface import ProductionSurfaceCache, ProductionSurfaceLiveState
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore
from tests.test_helpers import write_rgb_png


class ProductionSurfaceTests(unittest.TestCase):
    def test_surface_cache_uses_saved_pack_colors_and_reuses_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            write_rgb_png(brush_root / "land" / "red.png", rgb=(220, 30, 20))
            catalog = BrushCatalog(brush_root)
            scan = catalog.scan()
            record = scan.active_records[0]
            records = {record.uid: record}

            topology = ProductionTopology(16)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("planet", layout)
            chunk_id = next(
                item.chunk_id
                for item in layout.records
                if any(cell_id >= 12 for cell_id in layout.chunk_cell_ids(item.chunk_id))
            )
            for cell_id in layout.chunk_cell_ids(chunk_id):
                if cell_id >= 12:
                    session.set_cell(cell_id, store, record.uid, record.relative_path, 0)
            store.save(session)

            cache = ProductionSurfaceCache(brush_root)
            first, image = cache.ensure(layout, visibility, session, store, records)
            self.assertTrue(first.generated)
            self.assertEqual((image.width, image.height, image.channels), (1024, 512, 3))
            bound = visibility.chunks[chunk_id]
            import math

            longitude = math.atan2(bound.center[2], bound.center[0])
            latitude = math.asin(bound.center[1])
            x = int(round((longitude + math.pi) / (2.0 * math.pi) * (image.width - 1)))
            y = int(round((math.pi / 2.0 - latitude) / math.pi * (image.height - 1)))
            offset = (y * image.width + x) * 3
            color = tuple(image.pixels[offset : offset + 3])
            self.assertNotEqual(color, EMPTY_RGB)
            self.assertGreater(color[0], color[1])

            second, reused = cache.ensure(layout, visibility, session, store, records)
            self.assertFalse(second.generated)
            self.assertEqual(second.signature, first.signature)
            self.assertEqual(reused.pixels, image.pixels)

            live = ProductionSurfaceLiveState(visibility, image)
            self.assertTrue(live.owned_pixels[chunk_id])
            owned_pixel = live.owned_pixels[chunk_id][0]
            before = tuple(image.pixels[owned_pixel * 3 : owned_pixel * 3 + 3])
            self.assertGreater(before[0], before[1])

            session.clear_cells(store, layout.chunk_cell_ids(chunk_id))
            region, updated = live.update_chunks(
                {chunk_id: tuple(session.loaded_chunks[chunk_id])},
                cache.local_colors(session, records),
            )
            self.assertIsNotNone(region)
            after = tuple(updated.pixels[owned_pixel * 3 : owned_pixel * 3 + 3])
            self.assertEqual(after, EMPTY_RGB)
            self.assertNotEqual(updated.pixels, image.pixels)


if __name__ == "__main__":
    unittest.main()
