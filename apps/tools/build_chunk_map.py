from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunk_layout import build_chunk_layout, write_chunk_layout_cache
from app.chunk_visibility import build_chunk_visibility_index, write_chunk_visibility_cache
from app.paths import ProjectPaths
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology, write_topology_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic spherical chunks and optional pack-backed map storage."
    )
    parser.add_argument("--frequency", type=int, default=16)
    parser.add_argument("--target-cells", type=int, default=256)
    parser.add_argument("--map-name", default="")
    parser.add_argument("--create-map", action="store_true")
    parser.add_argument("--verify-map", action="store_true")
    parser.add_argument("--analyze-pack", action="store_true")
    parser.add_argument("--compact-pack", action="store_true")
    parser.add_argument("--recover-pack", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    expected_cells = 10 * args.frequency * args.frequency + 2
    max_cells = None if args.allow_large else 200_000

    topology = generate_dual_topology(args.frequency, max_cells=max_cells)
    topology_cache = write_topology_cache(topology, paths.map_root)
    layout = build_chunk_layout(topology, target_cells=args.target_cells)
    layout_cache = write_chunk_layout_cache(layout, paths.map_root / ".topology")
    visibility = build_chunk_visibility_index(topology, layout)
    visibility_cache = write_chunk_visibility_cache(visibility, paths.map_root / ".topology")
    validation = layout.validate(topology)
    sizes = [len(chunk.cell_ids) for chunk in layout.chunks]

    print(f"frequency={args.frequency}")
    print(f"cells={expected_cells}")
    print(f"chunks={layout.chunk_count}")
    print(f"chunk_min={min(sizes)}")
    print(f"chunk_average={sum(sizes) / len(sizes):.3f}")
    print(f"chunk_max={max(sizes)}")
    print(f"connected={validation.connected_chunks}")
    print(f"topology_hash={layout.topology_hash}")
    print(f"layout_hash={layout.stable_hash}")
    print(f"topology_cache={topology_cache.directory}")
    print(f"chunk_index={layout_cache.index_path}")
    print(f"visibility_nodes={visibility.node_count}")
    print(f"visibility_hash={visibility.stable_hash}")
    print(f"visibility_index={visibility_cache.index_path}")

    if (
        args.create_map
        or args.verify_map
        or args.analyze_pack
        or args.compact_pack
        or args.recover_pack
    ):
        map_name = args.map_name.strip() or f"planet_chunk_f{args.frequency}"
        store = SphereMapStore(paths.map_root)
        if map_name in store.list_maps():
            if args.recover_pack:
                state = store.compaction_recovery_state(map_name)
                if state.required:
                    recovery = store.recover_compaction(map_name, layout)
                    print(f"recovery_action={recovery.action}")
                    print(f"recovery_verified_chunks={recovery.verified_chunks}")
            session = store.open(map_name, layout)
            print(f"map_opened={session.map_dir}")
        elif args.create_map:
            session = store.create_blank(map_name, layout)
            print(f"map_created={session.map_dir}")
        else:
            print(f"map_error=map does not exist: {map_name}")
            return 3
        if args.recover_pack:
            state = store.compaction_recovery_state(map_name)
            print(f"recovery_required={state.required}")
            print(f"recovery_phase={state.phase}")
        if args.analyze_pack or args.compact_pack:
            analysis = store.analyze_storage(session)
            print(f"pack_physical_bytes={analysis.physical_bytes}")
            print(f"pack_live_bytes={analysis.live_bytes}")
            print(f"pack_reclaimable_bytes={analysis.reclaimable_bytes}")
            print(f"pack_physical_blocks={analysis.physical_blocks}")
            print(f"pack_orphan_blocks={analysis.orphan_blocks}")
            for issue in analysis.issues:
                print(f"pack_issue={issue}")
        if args.compact_pack:
            compacted = store.compact(session)
            print(f"pack_bytes_reclaimed={compacted.bytes_reclaimed}")
            print(f"pack_after_bytes={compacted.after.physical_bytes}")
            print(f"pack_after_orphan_blocks={compacted.after.orphan_blocks}")
            print(f"pack_compaction_verified_chunks={compacted.verified_chunks}")
        if args.verify_map:
            report = store.verify(session)
            print(f"map_valid={report.valid}")
            print(f"checked_chunks={report.checked_chunks}")
            if report.issues:
                for issue in report.issues:
                    print(f"issue={issue}")
            return 0 if report.valid else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
