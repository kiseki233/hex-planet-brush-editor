"""Check the locale catalogs against the ``t(...)`` calls in the source.

The Chinese source text is the lookup key, so the catalog can drift in two
directions: a new string appears in the UI and no translation exists, or a
string is reworded and the old entry becomes dead weight. Both are reported
here.

    python apps/tools/extract_messages.py            # report
    python apps/tools/extract_messages.py --template # write *.missing.json

A missing entry is not an error at runtime -- ``t()`` falls back to the Chinese
source -- so this is a maintenance aid, not a build gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = APPS_ROOT / "app"
LOCALE_DIR = APP_DIR / "locales"
SKIP = {"i18n.py"}


def collect_messages(app_dir: Path) -> dict[str, list[str]]:
    """Map every translatable string to the files that use it."""
    found: dict[str, list[str]] = {}
    for path in sorted(app_dir.glob("*.py")):
        if path.name in SKIP:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "t" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(path.name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true", help="write <code>.missing.json")
    parser.add_argument("--app", type=Path, default=APP_DIR)
    parser.add_argument("--locales", type=Path, default=LOCALE_DIR)
    args = parser.parse_args()

    messages = collect_messages(args.app)
    print(f"source strings: {len(messages)}")

    exit_code = 0
    for path in sorted(args.locales.glob("*.json")):
        if path.name.endswith(".missing.json"):
            continue
        catalog = json.loads(path.read_text(encoding="utf-8"))
        missing = [text for text in messages if text not in catalog]
        stale = [text for text in catalog if text not in messages]
        print(f"  {path.stem:4} translated {len(catalog):4}  missing {len(missing):4}  stale {len(stale):4}")
        for text in stale[:5]:
            print(f"        stale: {text[:60]!r}")
        if args.template and missing:
            target = path.with_name(f"{path.stem}.missing.json")
            target.write_text(
                json.dumps({text: "" for text in missing}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"        wrote {target.name}")
        if stale:
            exit_code = 0  # reported, not fatal
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
