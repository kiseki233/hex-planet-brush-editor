from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from app.production_layout import (
    ProductionChunkLayout,
    load_production_layout_cache,
    write_production_layout_cache,
)
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology


class ProductionTopologyTests(unittest.TestCase):
    def test_v1_cell_ids_centers_and_neighbors_match_materialized_topology(self) -> None:
        for frequency in (1, 2, 3, 4, 8):
            legacy = generate_dual_topology(frequency)
            production = ProductionTopology(frequency)
            self.assertEqual(production.cell_count, legacy.cell_count)
            self.assertEqual(production.pentagon_ids, legacy.pentagon_ids)
            for cell_id in range(legacy.cell_count):
                for first, second in zip(
                    production.cell_centers[cell_id], legacy.cell_centers[cell_id]
                ):
                    self.assertAlmostEqual(first, second, places=14)
                self.assertEqual(production.neighbors[cell_id], legacy.neighbors[cell_id])

    def test_procedural_dual_corners_match_materialized_triangle_centers(self) -> None:
        legacy = generate_dual_topology(8)
        production = ProductionTopology(8)
        for cell_id in range(legacy.cell_count):
            expected = [legacy.triangle_centers[item] for item in legacy.incident_triangles[cell_id]]
            actual = list(production.cell_corners(cell_id))
            self.assertEqual(len(actual), len(expected))
            for point in actual:
                nearest = min(
                    math.sqrt(sum((a - b) ** 2 for a, b in zip(point, other)))
                    for other in expected
                )
                self.assertLess(nearest, 1e-12)

    def test_frequency_1004_is_created_without_materialized_cell_arrays(self) -> None:
        topology = ProductionTopology(1004)
        self.assertEqual(topology.cell_count, 10_080_162)
        self.assertEqual(topology.triangle_count, 20_160_320)
        self.assertEqual(topology.hexagon_count, 10_080_150)
        self.assertNotIsInstance(topology.cell_centers, tuple)
        self.assertNotIsInstance(topology.neighbors, tuple)
        rng = random.Random(1004)
        samples = [*range(12), *(rng.randrange(12, topology.cell_count) for _ in range(1000))]
        validation = topology.validate(samples)
        self.assertTrue(validation.valid, validation.issues)
        for cell_id in samples[:100]:
            address = topology.owner_address(cell_id)
            self.assertEqual(
                topology.point_id(address.face_id, *address.weights), cell_id
            )

    def test_production_layout_is_connected_and_round_trips(self) -> None:
        topology = ProductionTopology(64)
        layout = ProductionChunkLayout(topology)
        validation = layout.validate(full_chunk_limit=10_000)
        self.assertTrue(validation.valid, validation.issues)
        self.assertGreaterEqual(validation.smallest_chunk, 128)
        self.assertLessEqual(validation.largest_chunk, 384)
        self.assertEqual(sum(record.cell_count for record in layout.records), topology.cell_count)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_production_layout_cache(layout, Path(temporary))
            loaded = load_production_layout_cache(info.directory)
            self.assertEqual(loaded.stable_hash, layout.stable_hash)
            self.assertEqual(loaded.records, layout.records)
            for cell_id in (0, 11, 12, 100, topology.cell_count - 1):
                self.assertEqual(loaded.chunk_for_cell(cell_id), layout.chunk_for_cell(cell_id))

    def test_production_layout_works_with_pack_map_store(self) -> None:
        topology = ProductionTopology(32)
        layout = ProductionChunkLayout(topology)
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("production", layout)
            self.assertEqual(len(session.index_records), layout.chunk_count)
            editable = next(cell_id for cell_id in range(12, topology.cell_count))
            session.set_cell(editable, store, "uid-production", "地形/test.png", 4)
            store.save(session)
            reopened = store.open("production", layout)
            uid, rotation = reopened.brush_uid_for_cell(editable, store)
            self.assertEqual(uid, "uid-production")
            self.assertEqual(rotation, 4)
            report = store.verify(reopened)
            self.assertTrue(report.valid, report.issues)


if __name__ == "__main__":
    unittest.main()
