from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.paths import ProjectPaths
from app.topology import TopologyError, generate_dual_topology, write_topology_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate an icosahedral dual topology cache.")
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow more than 200,000 cells in the in-memory Python prototype.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_cells = 10 * args.frequency * args.frequency + 2
    if expected_cells > 200_000 and not args.allow_large:
        print(
            f"Refusing to allocate {expected_cells:,} cells in the Python prototype. "
            "Use --allow-large only after checking available memory.",
            file=sys.stderr,
        )
        return 2

    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    try:
        topology = generate_dual_topology(args.frequency)
        info = write_topology_cache(topology, paths.map_root)
    except (TopologyError, OSError, MemoryError) as exc:
        print(f"Topology build failed: {exc}", file=sys.stderr)
        return 1

    validation = topology.validate()
    print(f"cache={info.directory}")
    print(f"cells={validation.cell_count}")
    print(f"triangles={validation.triangle_count}")
    print(f"edges={validation.edge_count}")
    print(f"pentagons={validation.pentagon_count}")
    print(f"hexagons={validation.hexagon_count}")
    print(f"euler={validation.euler_characteristic}")
    print(f"stable_hash={validation.stable_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
