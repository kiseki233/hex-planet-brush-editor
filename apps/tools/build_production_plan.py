from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.paths import ProjectPaths
from app.production_layout import (
    ProductionChunkLayout,
    load_production_layout_cache,
    write_production_layout_cache,
)
from app.production_topology import ProductionTopology
from app.production_visibility import (
    build_production_visibility_index,
    load_production_visibility_cache,
    write_production_visibility_cache,
)
from app.sphere_map_store import SphereMapStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the procedural production topology/layout without materializing all cells."
    )
    parser.add_argument("--frequency", type=int, default=1004)
    parser.add_argument("--tile-side", type=int, default=16)
    parser.add_argument("--map-name", default="")
    parser.add_argument("--create-map", action="store_true")
    parser.add_argument("--verify-map", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=4096)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    cache_dir = paths.map_root / ".topology" / f"ico_dual_f{args.frequency}_v2"

    started = time.perf_counter()
    if cache_dir.exists() and not args.force_rebuild:
        layout = load_production_layout_cache(cache_dir)
        source = "cache"
    else:
        topology = ProductionTopology(args.frequency)
        layout = ProductionChunkLayout(topology, tile_side=args.tile_side)
        write_production_layout_cache(layout, paths.map_root / ".topology")
        source = "generated"
    layout_seconds = time.perf_counter() - started
    topology = layout.topology

    visibility_started = time.perf_counter()
    visibility_manifest = cache_dir / "production_visibility.json"
    if visibility_manifest.exists() and not args.force_rebuild:
        visibility = load_production_visibility_cache(cache_dir, layout)
        visibility_source = "cache"
    else:
        visibility = build_production_visibility_index(layout)
        write_production_visibility_cache(visibility, paths.map_root / ".topology")
        visibility_source = "generated"
    visibility_seconds = time.perf_counter() - visibility_started

    sample_count = max(0, int(args.validate_samples))
    rng = random.Random(0x4858504C)
    sample_ids = list(range(12))
    if sample_count:
        sample_ids.extend(rng.randrange(12, topology.cell_count) for _ in range(sample_count))
    topology_validation = topology.validate(sample_ids)
    layout_validation = layout.validate(sample_chunks=min(512, max(1, layout.chunk_count)))
    visibility_validation = visibility.validate(layout)

    sizes = [record.cell_count for record in layout.records]
    print(f"layout_source={source}")
    print(f"visibility_source={visibility_source}")
    print(f"frequency={topology.frequency}")
    print(f"cells={topology.cell_count}")
    print(f"triangles={topology.triangle_count}")
    print(f"pentagons=12")
    print(f"hexagons={topology.hexagon_count}")
    print(f"chunks={layout.chunk_count}")
    print(f"chunk_min={min(sizes)}")
    print(f"chunk_average={sum(sizes) / len(sizes):.3f}")
    print(f"chunk_max={max(sizes)}")
    print(f"topology_hash={topology.stable_hash}")
    print(f"layout_hash={layout.stable_hash}")
    print(f"visibility_hash={visibility.stable_hash}")
    print(f"visibility_nodes={visibility.node_count}")
    print(f"visibility_leaves={visibility_validation.leaf_count}")
    print(f"procedural_cell_records=0")
    print(f"procedural_triangle_records=0")
    print(f"topology_sample_valid={topology_validation.valid}")
    print(f"topology_sample_checked={topology_validation.checked_cells}")
    print(f"layout_sample_valid={layout_validation.valid}")
    print(f"layout_chunks_checked={layout_validation.checked_chunks}")
    print(f"visibility_valid={visibility_validation.valid}")
    print(f"layout_seconds={layout_seconds:.6f}")
    print(f"visibility_seconds={visibility_seconds:.6f}")
    print(f"build_seconds={layout_seconds + visibility_seconds:.6f}")
    print(f"cache={cache_dir}")
    for issue in (
        *topology_validation.issues,
        *layout_validation.issues,
        *visibility_validation.issues,
    ):
        print(f"issue={issue}")

    if (
        not topology_validation.valid
        or not layout_validation.valid
        or not visibility_validation.valid
    ):
        return 2

    if args.create_map or args.verify_map:
        map_name = args.map_name.strip() or f"planet_production_f{args.frequency}"
        store = SphereMapStore(paths.map_root)
        if map_name in store.list_maps():
            session = store.open(map_name, layout)
            print(f"map_opened={session.map_dir}")
        elif args.create_map:
            create_started = time.perf_counter()
            session = store.create_blank(map_name, layout)
            print(f"map_created={session.map_dir}")
            print(f"map_create_seconds={time.perf_counter() - create_started:.6f}")
        else:
            print(f"map_error=map does not exist: {map_name}")
            return 3
        if args.verify_map:
            verify_started = time.perf_counter()
            report = store.verify(session)
            print(f"map_valid={report.valid}")
            print(f"checked_chunks={report.checked_chunks}")
            print(f"verify_seconds={time.perf_counter() - verify_started:.6f}")
            for issue in report.issues:
                print(f"map_issue={issue}")
            return 0 if report.valid else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
