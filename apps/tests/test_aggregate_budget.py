from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.aggregate_lod import AggregateLodCache, ProductionAggregateHierarchy
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore


class AggregateBudgetTest(unittest.TestCase):
    """A far view must not build every selected node inline.

    Building one node summarises every Pack chunk beneath it. A whole-planet view
    selects several hundred, which measured at up to 61 s for a single wheel notch
    before the budget existed.
    """

    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)
        cls.visibility = build_production_visibility_index(cls.layout)
        cls.hierarchy = ProductionAggregateHierarchy(cls.visibility, cls.layout)

    def _session(self, temporary):
        store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
        return store, store.create_blank("planet", self.layout)

    def _selection(self):
        return tuple(range(min(20, self.hierarchy.node_count)))

    def test_budget_caps_generation_and_reports_the_rest(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            nodes = self._selection()
            report = cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, nodes,
                generate_budget=3,
            )
            self.assertEqual(report.generated_count, 3)
            self.assertEqual(report.pending_count, len(nodes) - 3)
            self.assertEqual(report.node_count, len(nodes))

    def test_repeated_budgeted_calls_converge(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            nodes = self._selection()
            for _ in range(50):
                report = cache.ensure_nodes(
                    self.hierarchy, self.layout, session, store, {}, nodes,
                    generate_budget=3,
                )
                if report.pending_count == 0:
                    break
            self.assertEqual(report.pending_count, 0)
            # One more pass: everything is now cached, so nothing is rebuilt.
            settled = cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, nodes,
                generate_budget=3,
            )
            self.assertEqual(settled.generated_count, 0)
            self.assertEqual(settled.reused_count, len(nodes))

    def test_no_budget_still_builds_everything(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            nodes = self._selection()
            report = cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, nodes
            )
            self.assertEqual(report.pending_count, 0)
            self.assertEqual(report.generated_count, len(nodes))


class ManifestMemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)
        cls.visibility = build_production_visibility_index(cls.layout)
        cls.hierarchy = ProductionAggregateHierarchy(cls.visibility, cls.layout)

    def test_memo_returns_the_same_summary_without_rereading(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
            session = store.create_blank("planet", self.layout)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, (0,)
            )
            first = cache.existing(session.map_dir, 0)
            second = cache.existing(session.map_dir, 0)
            self.assertIsNotNone(first)
            self.assertIs(first, second)

    def test_invalidated_node_is_dropped_from_the_memo(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
            session = store.create_blank("planet", self.layout)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, (0,)
            )
            self.assertIsNotNone(cache.existing(session.map_dir, 0))
            cache.manifest_path(session.map_dir, 0).unlink()
            self.assertIsNone(cache.existing(session.map_dir, 0))

    def test_missing_image_invalidates_the_memo(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
            session = store.create_blank("planet", self.layout)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            cache.ensure_nodes(
                self.hierarchy, self.layout, session, store, {}, (0,)
            )
            self.assertIsNotNone(cache.existing(session.map_dir, 0))
            cache.image_path(session.map_dir, 0).unlink()
            self.assertIsNone(cache.existing(session.map_dir, 0))


if __name__ == "__main__":
    unittest.main()
