from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[1]
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from app.aggregate_lod import AggregateLodCache, ProductionAggregateHierarchy
from app.brush_catalog import BrushCatalog
from app.paths import ProjectPaths
from app.production_layout import load_production_layout_cache
from app.production_visibility import load_production_visibility_cache
from app.sphere_map_store import SphereMapStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build production aggregate LOD caches")
    parser.add_argument("--frequency", type=int, default=1004)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--yaw", type=float, default=-0.35)
    parser.add_argument("--pitch", type=float, default=0.25)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--target-pixels", type=float, default=96.0)
    parser.add_argument("--all", action="store_true", help="Build every hierarchy node")
    args = parser.parse_args()

    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    cache_dir = paths.map_root / ".topology" / f"ico_dual_f{args.frequency}_v2"
    layout = load_production_layout_cache(cache_dir)
    visibility = load_production_visibility_cache(cache_dir, layout)
    store = SphereMapStore(paths.map_root)
    session = store.open(args.map_name, layout)
    if session.dirty_chunks or session.brush_table_dirty:
        raise RuntimeError("Map has unsaved changes")
    records = {
        record.uid: record for record in BrushCatalog(paths.brush_root).scan().records
    }
    hierarchy = ProductionAggregateHierarchy(visibility, layout)
    lod_cache = AggregateLodCache(paths.brush_root)
    started = time.perf_counter()
    if args.all:
        report = lod_cache.build_all(hierarchy, layout, session, store, records)
        selected = report.node_count
        represented = layout.chunk_count
    else:
        selection = hierarchy.select(
            args.yaw,
            args.pitch,
            args.zoom,
            args.width,
            args.height,
            target_pixels=args.target_pixels,
        )
        report = lod_cache.ensure_nodes(
            hierarchy, layout, session, store, records, selection.node_ids
        )
        selected = len(selection.node_ids)
        represented = selection.represented_chunks
    print(f"map={session.name}")
    print(f"hierarchy_nodes={hierarchy.node_count}")
    print(f"selected_nodes={selected}")
    print(f"represented_chunks={represented}")
    print(f"generated={report.generated_count}")
    print(f"reused={report.reused_count}")
    print(f"cells={report.total_cells}")
    print(f"seconds={time.perf_counter() - started:.6f}")
    print(f"cache={lod_cache.root(session.map_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
