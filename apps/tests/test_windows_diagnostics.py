from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.paths import ProjectPaths
from app.windows_diagnostics import collect_windows_diagnostics


class WindowsDiagnosticsTests(unittest.TestCase):
    def _paths(self, root: Path) -> ProjectPaths:
        apps = root / "apps"
        art = root / "art"
        (apps / "runtime" / "logs").mkdir(parents=True)
        (art / "brushes").mkdir(parents=True)
        (art / "maps").mkdir(parents=True)
        (apps / "start.bat").write_text("@echo off\n", encoding="utf-8")
        (apps / "start.ps1").write_text("Write-Host ok\n", encoding="utf-8")
        return ProjectPaths(
            project_root=root,
            apps_root=apps,
            art_root=art,
            brush_root=art / "brushes",
            map_root=art / "maps",
            runtime_root=apps / "runtime",
            log_root=apps / "runtime" / "logs",
        )

    def test_diagnostic_writes_machine_readable_and_text_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            report = collect_windows_diagnostics(paths, run_wgl_probe=False)
            json_path, text_path = report.write(paths.log_root)
            self.assertTrue(json_path.is_file())
            self.assertTrue(text_path.is_file())
            self.assertIn("Launch scripts", text_path.read_text(encoding="utf-8"))
            self.assertEqual(report.failed_count, 0)
            if os.name != "nt":
                self.assertGreaterEqual(report.needs_windows_count, 1)
                self.assertFalse(bool(report.gpu.get("executed")))


if __name__ == "__main__":
    unittest.main()
