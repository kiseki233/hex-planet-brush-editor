from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.production_visibility import (
    ProductionVisibilityError,
    build_production_visibility_index,
    load_production_visibility_cache,
    write_production_visibility_cache,
)


class ProductionVisibilityTests(unittest.TestCase):
    def test_bounds_contain_every_cell_at_frequency_128(self) -> None:
        topology = ProductionTopology(128)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout)
        for record, bound in zip(layout.records, visibility.chunks):
            for cell_id in layout.chunk_cell_ids(record.chunk_id):
                center = topology.cell_center(cell_id)
                dot = sum(a * b for a, b in zip(bound.center, center))
                angle = math.acos(max(-1.0, min(1.0, dot)))
                self.assertLessEqual(angle, bound.angular_radius + 1e-10)

    def test_cache_round_trip_and_corruption_detection(self) -> None:
        topology = ProductionTopology(64)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_production_visibility_cache(visibility, temporary)
            loaded = load_production_visibility_cache(info.directory, layout)
            self.assertEqual(loaded.stable_hash, visibility.stable_hash)
            self.assertEqual(loaded.chunks, visibility.chunks)
            payload = bytearray(info.index_path.read_bytes())
            payload[len(payload) // 2] ^= 0x5A
            info.index_path.write_bytes(payload)
            with self.assertRaises(ProductionVisibilityError):
                load_production_visibility_cache(info.directory, layout)

    def test_large_query_is_compact_and_prunes(self) -> None:
        topology = ProductionTopology(256)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout)
        query = visibility.query(layout, -0.35, 0.25, 12.0, 1100, 760)
        self.assertEqual(ProductionTopology(1004).cell_count, 10_080_162)
        self.assertLess(visibility.node_count, layout.chunk_count)
        self.assertLess(len(query.chunk_ids), layout.chunk_count // 8)
        self.assertLess(query.candidate_cells, topology.cell_count // 20)
        self.assertLess(query.tested_chunks, layout.chunk_count // 3)


if __name__ == "__main__":
    unittest.main()
