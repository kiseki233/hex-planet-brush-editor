"""Generate the placeholder brush library shipped with this repository.

The editor itself carries no artwork. This script writes a small set of flat
colour tiles so that a fresh clone opens with a usable brush library and the
whole pipeline -- catalog scan, LOD cache, GPU streaming, painting -- can be
exercised without supplying any assets first.

Replace ``art/brushes`` with your own 512x512 PNG tiles whenever you want real
artwork. The catalog is rebuilt from whatever is on disk, so no registration
step is needed.

Usage:
    python apps/tools/make_example_brushes.py [--size 512] [--variants 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parent.parent
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from app.png_pixels import PixelImage, atomic_write_png  # noqa: E402

PROJECT_ROOT = APPS_ROOT.parent

# Flat colours chosen to read as distinct terrain families at a glance.
# They are plain constants, not derived from any external imagery.
CATEGORIES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("01-ocean-deep", (26, 58, 110)),
    ("02-ocean-shallow", (52, 120, 170)),
    ("10-city-core", (88, 88, 94)),
    ("11-city-suburb", (134, 130, 122)),
    ("16-farmland", (168, 168, 96)),
    ("18-plains", (120, 150, 86)),
    ("25-mountain", (110, 100, 84)),
    ("32-snow-peak", (225, 228, 232)),
    ("44-desert", (216, 188, 130)),
    ("51-grassland", (150, 170, 90)),
    ("54-forest", (58, 110, 62)),
    ("73-snowfield", (236, 240, 244)),
)

# Per-variant brightness so each category holds several distinguishable tiles.
VARIANT_FACTORS: tuple[float, ...] = (0.82, 0.94, 1.06, 1.18)


def shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in rgb)  # type: ignore[return-value]


def solid_tile(rgb: tuple[int, int, int], size: int) -> PixelImage:
    pixel = bytes((rgb[0], rgb[1], rgb[2], 255))
    return PixelImage(width=size, height=size, channels=4, pixels=pixel * (size * size))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="tile edge length in pixels")
    parser.add_argument(
        "--variants",
        type=int,
        default=len(VARIANT_FACTORS),
        help=f"tiles per category, 1..{len(VARIANT_FACTORS)}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "art" / "brushes",
        help="brush library root",
    )
    args = parser.parse_args()

    if args.size < 8:
        parser.error("--size must be at least 8")
    if not 1 <= args.variants <= len(VARIANT_FACTORS):
        parser.error(f"--variants must be between 1 and {len(VARIANT_FACTORS)}")

    written = 0
    for name, base_rgb in CATEGORIES:
        category_dir = args.out / name
        category_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.variants):
            rgb = shade(base_rgb, VARIANT_FACTORS[index])
            target = category_dir / f"{index + 1:02d}.png"
            atomic_write_png(target, solid_tile(rgb, args.size), durable=False)
            written += 1

    print(f"wrote {written} tiles across {len(CATEGORIES)} categories into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
