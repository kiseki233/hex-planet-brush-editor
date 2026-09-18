from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.gpu_stream import VisibleGpuInstanceStream
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from tests.test_helpers import write_rgb_png


class GpuStreamResidencyIntegrationTests(unittest.TestCase):
    def test_hidden_brush_layer_is_released_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            write_rgb_png(brush_root / "terrain" / "a.png", rgb=(200, 20, 20))
            write_rgb_png(brush_root / "terrain" / "b.png", rgb=(20, 20, 200))
            scan = BrushCatalog(brush_root).scan()
            records = {record.uid: record for record in scan.active_records}
            by_name = {Path(record.relative_path).name: record for record in scan.active_records}
            topology = ProductionTopology(16)
            layout = ProductionChunkLayout(topology)
            chunks = [
                chunk_id
                for chunk_id in range(layout.chunk_count)
                if any(cell_id >= 12 for cell_id in layout.chunks[chunk_id].cell_ids)
            ][:2]
            self.assertEqual(len(chunks), 2)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("sphere", layout)
            cells = [next(cell for cell in layout.chunks[c].cell_ids if cell >= 12) for c in chunks]
            for cell, record in zip(cells, (by_name["a.png"], by_name["b.png"])):
                session.set_cell(cell, store, record.uid, record.relative_path, 0)
            store.save(session)
            session.loaded_chunks.clear()

            stream = VisibleGpuInstanceStream(brush_root, texture_grace_ticks=0)
            first = stream.update_visible_chunks(topology, layout, session, store, records, [chunks[0]])
            stream.drain_texture_events()
            layer_a = stream.residency.layer_for_key(next(
                key for key in stream.layer_by_key if key not in {"__empty__", "__missing__"}
            ))
            self.assertIsNotNone(layer_a)

            second = stream.update_visible_chunks(topology, layout, session, store, records, [chunks[1]])
            self.assertIn(layer_a, second.released_texture_layers)
            stream.drain_texture_events()

            third = stream.update_visible_chunks(topology, layout, session, store, records, [chunks[0]])
            events = stream.drain_texture_events()
            self.assertTrue(any(event.replace_existing for event in events))
            active_brush_layers = [
                slot.layer for slot in stream.residency.active_slots() if not slot.pinned
            ]
            self.assertEqual(active_brush_layers, [layer_a])
            self.assertIn(3, third.released_texture_layers)


if __name__ == "__main__":
    unittest.main()
