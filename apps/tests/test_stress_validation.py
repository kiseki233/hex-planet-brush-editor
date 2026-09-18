from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.stress_validation import run_texture_stress


class StressValidationTests(unittest.TestCase):
    def test_texture_stress_is_bounded_and_reuses_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = run_texture_stress(Path(temporary), brush_count=12, lod_level=1)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.unique_texture_keys, 12)
            self.assertEqual(report.logical_rgba_bytes, report.expected_rgba_bytes)
            self.assertEqual(report.released_layers, 6)
            self.assertEqual(report.reused_layers, 6)


if __name__ == "__main__":
    unittest.main()
