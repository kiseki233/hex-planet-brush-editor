from __future__ import annotations

import random
import unittest

from app.brush_stroke import BrushStrokePlanner, BrushStrokeState, BrushStrokeTool
from app.production_topology import ProductionTopology


class ReferencePlanner(BrushStrokePlanner):
    """The pre-incremental algorithm: re-dilate the whole disc every segment."""

    def plan_segment(self, state, current_cell_id):
        current_cell_id = int(current_cell_id)
        if state.last_cell_id is None:
            centerline = (current_cell_id,)
        else:
            centerline = self.connect(state.last_cell_id, current_cell_id)
        affected = self._dilate(centerline, state.tool.diameter)
        fresh = tuple(
            cell_id
            for cell_id in affected
            if cell_id not in state.painted_cell_ids and cell_id not in self._pentagons
        )
        state.painted_cell_ids.update(fresh)
        state.last_cell_id = current_cell_id
        state.touched_cell_count += len(fresh)
        from app.brush_stroke import BrushStrokePlan

        return BrushStrokePlan(centerline=centerline, affected_cell_ids=fresh)


class IncrementalEquivalenceTest(unittest.TestCase):
    """The incremental expansion must cover exactly what re-dilation covered."""

    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(32)
        cls.planner = BrushStrokePlanner(cls.topology)
        cls.reference = ReferencePlanner(cls.topology)

    def _walk(self, planner, start, steps, diameter, seed):
        rng = random.Random(seed)
        state = BrushStrokeState(BrushStrokeTool("erase", diameter))
        cell = start
        per_segment = []
        for _ in range(steps):
            plan = planner.plan_segment(state, cell)
            per_segment.append(frozenset(plan.affected_cell_ids))
            neighbors = self.topology.cell_neighbor_ids_unordered(cell)
            cell = neighbors[rng.randrange(len(neighbors))]
        return state, per_segment

    def test_single_dab_matches(self):
        for diameter in (1, 2, 5, 16, 41):
            with self.subTest(diameter=diameter):
                mine, _ = self._walk(self.planner, 900, 1, diameter, 1)
                theirs, _ = self._walk(self.reference, 900, 1, diameter, 1)
                self.assertEqual(mine.painted_cell_ids, theirs.painted_cell_ids)

    def test_random_walk_matches_segment_by_segment(self):
        for diameter in (1, 3, 9, 24):
            for seed in (2, 17, 99):
                with self.subTest(diameter=diameter, seed=seed):
                    mine, mine_segments = self._walk(
                        self.planner, 1500, 12, diameter, seed
                    )
                    theirs, their_segments = self._walk(
                        self.reference, 1500, 12, diameter, seed
                    )
                    self.assertEqual(mine.painted_cell_ids, theirs.painted_cell_ids)
                    self.assertEqual(mine_segments, their_segments)
                    self.assertEqual(mine.touched_cell_count, theirs.touched_cell_count)

    def test_long_jump_matches(self):
        """A fast pointer move connects a long path before dilating."""
        mine = BrushStrokeState(BrushStrokeTool("erase", 11))
        theirs = BrushStrokeState(BrushStrokeTool("erase", 11))
        for cell in (400, 4000, 900, 7000):
            self.planner.plan_segment(mine, cell)
            self.reference.plan_segment(theirs, cell)
        self.assertEqual(mine.painted_cell_ids, theirs.painted_cell_ids)

    def test_pentagons_are_traversed_but_never_painted(self):
        state = BrushStrokeState(BrushStrokeTool("erase", 60))
        plan = self.planner.plan_segment(state, 0)
        self.assertFalse(set(plan.affected_cell_ids) & set(range(12)))
        # The five neighbours of the pentagon are still reached through it.
        for neighbor_id in self.topology.cell_neighbor_ids_unordered(0):
            self.assertIn(neighbor_id, state.painted_cell_ids)

    def test_distance_map_never_exceeds_the_radius(self):
        state = BrushStrokeState(BrushStrokeTool("erase", 21))
        self.planner.plan_segment(state, 2000)
        radius = BrushStrokePlanner.graph_radius(21)
        self.assertTrue(state.distance_to_path)
        self.assertLessEqual(max(state.distance_to_path.values()), radius)

    def test_repeat_at_the_same_cell_adds_nothing(self):
        state = BrushStrokeState(BrushStrokeTool("erase", 15))
        first = self.planner.plan_segment(state, 1200)
        second = self.planner.plan_segment(state, 1200)
        self.assertGreater(len(first.affected_cell_ids), 1)
        self.assertEqual(second.affected_cell_ids, ())


class IncrementalCostTest(unittest.TestCase):
    """A pointer move must cost the crescent, not the whole disc."""

    def test_later_moves_expand_far_fewer_cells(self):
        topology = ProductionTopology(64)
        planner = BrushStrokePlanner(topology)
        state = BrushStrokeState(BrushStrokeTool("erase", 61))
        cell = 20_000

        first = planner.plan_segment(state, cell)
        reached_after_first = len(state.distance_to_path)

        for _ in range(3):
            cell = topology.cell_neighbor_ids_unordered(cell)[0]
        before = len(state.distance_to_path)
        second = planner.plan_segment(state, cell)
        grew_by = len(state.distance_to_path) - before

        self.assertGreater(len(first.affected_cell_ids), 2000)
        self.assertLess(len(second.affected_cell_ids), 400)
        # The distance map may only grow by roughly the new crescent.
        self.assertLess(grew_by, reached_after_first // 4)


if __name__ == "__main__":
    unittest.main()
