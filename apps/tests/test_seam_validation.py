from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.production_topology import ProductionTopology
from app.seam_validation import create_seam_test_brushes, validate_topology_seams


class SeamValidationTests(unittest.TestCase):
    def test_procedural_shared_edges_and_uv_padding(self) -> None:
        topology = ProductionTopology(32)
        report = validate_topology_seams(topology, sample_limit=256)
        self.assertTrue(report.valid, report.issues[:3])
        self.assertGreater(report.checked_neighbor_pairs, 100)
        self.assertLess(report.max_shared_corner_error, 1e-10)
        self.assertTrue(report.uv_inside_padded_unit)

    def test_seam_test_brushes_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_seam_test_brushes(Path(temporary))
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
