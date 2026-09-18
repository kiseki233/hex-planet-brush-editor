from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.topology import (
    CELL_HEADER,
    CELL_RECORD,
    TRIANGLE_HEADER,
    TRIANGLE_RECORD,
    generate_dual_topology,
    verify_topology_cache,
    write_topology_cache,
)


class TopologyTests(unittest.TestCase):
    def test_expected_counts_and_euler_characteristic(self) -> None:
        for frequency in (1, 2, 3, 4, 8):
            topology = generate_dual_topology(frequency)
            validation = topology.validate()
            self.assertTrue(validation.valid)
            self.assertEqual(validation.cell_count, 10 * frequency * frequency + 2)
            self.assertEqual(validation.triangle_count, 20 * frequency * frequency)
            self.assertEqual(validation.edge_count, 30 * frequency * frequency)
            self.assertEqual(validation.euler_characteristic, 2)

    def test_exactly_twelve_original_vertices_are_pentagons(self) -> None:
        topology = generate_dual_topology(8)
        self.assertEqual(topology.pentagon_ids, tuple(range(12)))
        self.assertTrue(all(len(topology.neighbors[cell_id]) == 5 for cell_id in range(12)))
        self.assertTrue(all(len(topology.neighbors[cell_id]) == 6 for cell_id in range(12, topology.cell_count)))

    def test_neighbors_are_reciprocal_and_unique(self) -> None:
        topology = generate_dual_topology(6)
        for cell_id, neighbors in enumerate(topology.neighbors):
            self.assertEqual(len(neighbors), len(set(neighbors)))
            self.assertNotIn(cell_id, neighbors)
            for neighbor_id in neighbors:
                self.assertIn(cell_id, topology.neighbors[neighbor_id])

    def test_cell_ids_and_hash_are_stable(self) -> None:
        first = generate_dual_topology(5)
        second = generate_dual_topology(5)
        self.assertEqual(first.cell_centers, second.cell_centers)
        self.assertEqual(first.triangles, second.triangles)
        self.assertEqual(first.neighbors, second.neighbors)
        self.assertEqual(first.stable_hash, second.stable_hash)

    def test_dual_corner_count_matches_cell_degree(self) -> None:
        topology = generate_dual_topology(7)
        for cell_id in range(topology.cell_count):
            self.assertEqual(len(topology.incident_triangles[cell_id]), len(topology.neighbors[cell_id]))

    def test_cache_round_trip_verification_and_record_sizes(self) -> None:
        topology = generate_dual_topology(4)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_topology_cache(topology, Path(temporary))
            result = verify_topology_cache(info.directory)
            self.assertTrue(result["valid"])

            manifest = json.loads(info.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["cells"], topology.cell_count)
            self.assertEqual(manifest["counts"]["pentagons"], 12)
            self.assertEqual(
                info.cells_path.stat().st_size,
                CELL_HEADER.size + topology.cell_count * CELL_RECORD.size,
            )
            self.assertEqual(
                info.triangles_path.stat().st_size,
                TRIANGLE_HEADER.size + topology.triangle_count * TRIANGLE_RECORD.size,
            )

    def test_cache_corruption_is_detected(self) -> None:
        topology = generate_dual_topology(2)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_topology_cache(topology, Path(temporary))
            payload = bytearray(info.cells_path.read_bytes())
            payload[-1] ^= 0xFF
            info.cells_path.write_bytes(payload)
            result = verify_topology_cache(info.directory)
            self.assertFalse(result["valid"])
            self.assertIn("checksum mismatch for cells", result["issues"])


if __name__ == "__main__":
    unittest.main()
