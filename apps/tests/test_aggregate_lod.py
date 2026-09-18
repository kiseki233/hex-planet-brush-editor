from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.aggregate_lod import (
    AggregateLodCache,
    ProductionAggregateHierarchy,
    build_aggregate_proxy_batch,
)
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore


class AggregateLodTests(unittest.TestCase):
    def test_hierarchy_selects_fewer_nodes_at_far_distance(self) -> None:
        topology = ProductionTopology(64)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout)
        hierarchy = ProductionAggregateHierarchy(visibility, layout)
        far = hierarchy.select(-0.2, 0.2, 1.0, 1100, 760, target_pixels=800)
        near = hierarchy.select(-0.2, 0.2, 10.0, 1100, 760, target_pixels=800)
        far_average = far.represented_chunks / len(far.node_ids)
        near_average = near.represented_chunks / len(near.node_ids)
        self.assertGreater(far_average, near_average)
        self.assertGreater(far.represented_chunks, 0)

    def test_selected_nodes_are_built_without_full_tree_prebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(32)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            hierarchy = ProductionAggregateHierarchy(visibility, layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("planet", layout)
            cache = AggregateLodCache(root / "brushes")
            selection = hierarchy.select(0.0, 0.0, 1.0, 900, 650, target_pixels=500)
            report = cache.ensure_nodes(
                hierarchy, layout, session, store, {}, selection.node_ids
            )
            self.assertEqual(report.node_count, len(selection.node_ids))
            self.assertEqual(report.generated_count, len(selection.node_ids))
            self.assertLess(report.node_count, hierarchy.node_count)
            second = cache.ensure_nodes(
                hierarchy, layout, session, store, {}, selection.node_ids
            )
            self.assertEqual(second.generated_count, 0)
            self.assertEqual(second.reused_count, len(selection.node_ids))

    def test_cache_build_reuse_invalidate_and_proxy_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(16)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout, leaf_size=4)
            hierarchy = ProductionAggregateHierarchy(visibility, layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("planet", layout)
            cache = AggregateLodCache(root / "brushes")
            first = cache.build_all(hierarchy, layout, session, store, {})
            self.assertEqual(first.total_cells, topology.cell_count)
            self.assertGreater(first.generated_count, 0)
            second = cache.build_all(hierarchy, layout, session, store, {})
            self.assertEqual(second.generated_count, 0)
            selected = hierarchy.select(0.0, 0.0, 1.0, 800, 600)
            batch = build_aggregate_proxy_batch(
                hierarchy,
                selected,
                topology.stable_hash,
                layout.stable_hash,
                session.map_dir,
                cache,
            )
            self.assertEqual(batch.instance_count, len(selected.node_ids))
            self.assertGreaterEqual(batch.texture_layer_count, 3)
            invalidated = cache.invalidate_chunks(hierarchy, session.map_dir, [0])
            self.assertTrue(invalidated)
            self.assertIsNone(cache.existing(session.map_dir, invalidated[0]))


if __name__ == "__main__":
    unittest.main()
