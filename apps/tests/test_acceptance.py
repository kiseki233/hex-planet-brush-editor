from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.acceptance import AcceptanceItem, AcceptanceReport


class AcceptanceReportTests(unittest.TestCase):
    def test_report_counts_and_writes_outputs(self) -> None:
        report = AcceptanceReport(
            version="v1.3.1",
            items=(
                AcceptanceItem("A", "pass", "ok", 0.1),
                AcceptanceItem("B", "needs_windows", "target PC", 0.2),
            ),
            started_utc=1,
            completed_utc=2,
        )
        self.assertTrue(report.valid)
        self.assertEqual(report.passed_count, 1)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.needs_windows_count, 1)
        with tempfile.TemporaryDirectory() as temporary:
            json_path, text_path = report.write(Path(temporary))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["needsWindowsCount"], 1)
            self.assertIn("Needs Windows: 1", text_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
