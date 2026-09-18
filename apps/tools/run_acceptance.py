from __future__ import annotations

import argparse
import sys
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[1]
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from app.acceptance import run_final_acceptance
from app.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hex Planet final acceptance")
    parser.add_argument("--quick-texture-stress", action="store_true")
    parser.add_argument("--skip-wgl-probe", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths.from_app_file(APPS_ROOT / "app" / "main.py")
    paths.ensure_required_directories()
    report = run_final_acceptance(
        paths,
        full_texture_stress=not args.quick_texture_stress,
        run_wgl_probe=not args.skip_wgl_probe,
    )
    json_path, text_path = report.write(paths.log_root)
    print(f"passed={report.passed_count}")
    print(f"failed={report.failed_count}")
    print(f"needs_windows={report.needs_windows_count}")
    print(f"json={json_path}")
    print(f"text={text_path}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
