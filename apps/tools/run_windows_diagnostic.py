from __future__ import annotations

import argparse
import sys
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[1]
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from app.paths import ProjectPaths
from app.windows_diagnostics import collect_windows_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hex Planet Windows/WGL diagnostics")
    parser.add_argument("--skip-wgl-probe", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    report = collect_windows_diagnostics(paths, run_wgl_probe=not args.skip_wgl_probe)
    json_path, text_path = report.write(paths.log_root)
    for item in report.items:
        print(f"[{item.status}] {item.name}: {item.detail}")
    print(f"gpu={report.gpu}")
    print(f"json={json_path}")
    print(f"text={text_path}")
    return 1 if report.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
