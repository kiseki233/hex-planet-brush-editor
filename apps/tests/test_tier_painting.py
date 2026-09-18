from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.brush_stroke import BrushStrokePlanner, BrushStrokeState, BrushStrokeTool
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from app.sphere_picker import SphereScreenPicker
from app.undo_history import UndoHistory
from app.zoom_tiers import ZOOM_TIERS, ZoomTierPolicy

WIDTH = HEIGHT = 900
CENTER_X = CENTER_Y = 450.0
YAW = -0.35
PITCH = 0.25


def sphere_radius(zoom: float) -> float:
    """Match ``ProductionSphereEditor._sphere_geometry``."""
    return min(WIDTH, HEIGHT) * 0.42 * zoom


class TierPaintingTest(unittest.TestCase):
    """Every tier must be able to complete a real stroke.

    This is the regression guard for the behaviour that used to be blocked: the
    editor only accepted a click once ``zoom >= 36`` had produced detail
    polygons, so five of the six tiers could not paint at all.
    """

    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)
        cls.picker = SphereScreenPicker(cls.topology)
        cls.planner = BrushStrokePlanner(cls.topology)

    def _fresh_session(self, temporary):
        store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
        return store, store.create_blank("planet", self.layout)

    def test_every_tier_resolves_a_cell_under_the_cursor(self):
        for tier in ZOOM_TIERS:
            for zoom in (tier.zoom_floor, tier.zoom_floor * 1.5):
                cell_id = self.picker.pick(
                    CENTER_X + 11.0,
                    CENTER_Y - 7.0,
                    CENTER_X,
                    CENTER_Y,
                    sphere_radius(zoom),
                    YAW,
                    PITCH,
                )
                self.assertIsNotNone(
                    cell_id, f"tier {tier.name} at zoom {zoom} produced no cell"
                )

    def test_every_tier_paints_and_undoes(self):
        for tier in ZOOM_TIERS:
            with self.subTest(tier=tier.name):
                with tempfile.TemporaryDirectory() as temporary:
                    store, session = self._fresh_session(temporary)
                    history = UndoHistory()
                    diameter = tier.clamp_diameter(tier.suggested_diameter)
                    tool = BrushStrokeTool(
                        "paint",
                        diameter,
                        (_FakeRecord("uid-a", "地形/a.png"),),
                    ).validated()

                    cell_id = self.picker.pick(
                        CENTER_X,
                        CENTER_Y,
                        CENTER_X,
                        CENTER_Y,
                        sphere_radius(tier.zoom_floor),
                        YAW,
                        PITCH,
                    )
                    self.assertIsNotNone(cell_id)

                    state = BrushStrokeState(tool=tool)
                    plan = self.planner.plan_segment(state, cell_id)
                    self.assertGreater(len(plan.affected_cell_ids), 0)

                    history.begin(f"{tier.name} 绘制")
                    touched_chunks = set()
                    for affected in plan.affected_cell_ids:
                        chunk_id, _ = self.layout.chunk_for_cell(affected)
                        if chunk_id in touched_chunks:
                            continue
                        touched_chunks.add(chunk_id)
                        history.capture(chunk_id, store.load_chunk(session, chunk_id))
                    changed = session.set_cells(
                        store,
                        {
                            affected: ("uid-a", "地形/a.png", 0)
                            for affected in plan.affected_cell_ids
                        },
                    )
                    history.commit(len(changed))
                    self.assertGreater(len(changed), 0)

                    result = history.undo(session, store, self.layout)
                    self.assertIsNotNone(result)
                    self.assertEqual(len(result.changed_cell_ids), len(changed))
                    for affected in changed:
                        self.assertEqual(session.get_state(affected, store), (0, 0))

    def test_diameter_clamp_keeps_far_tier_strokes_visible(self):
        """A far tier must not accept a brush finer than its own feedback."""
        policy = ZoomTierPolicy(initial_zoom=1.0)
        self.assertEqual(policy.tier.index, 5)
        self.assertGreaterEqual(policy.clamp_diameter(1), policy.tier.feedback_cells)

    def test_near_tier_brush_stays_inside_the_viewport(self):
        """At maximum zoom the brush must not exceed a couple of screens."""
        near = ZOOM_TIERS[0]
        # 969 / zoom is roughly the cell radius the window spans.
        window_cells = 969.0 / near.zoom_floor
        self.assertLessEqual(near.max_diameter / 2.0, window_cells * 2.5)

    def test_undo_brush_reverts_one_change_per_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._fresh_session(temporary)
            history = UndoHistory()
            cell_id = self.picker.pick(
                CENTER_X, CENTER_Y, CENTER_X, CENTER_Y, sphere_radius(1.0), YAW, PITCH
            )
            state = BrushStrokeState(tool=BrushStrokeTool("erase", 9).validated())
            plan = self.planner.plan_segment(state, cell_id)
            cells = [item for item in plan.affected_cell_ids if item >= 12]
            self.assertGreater(len(cells), 3)

            history.begin("绘制")
            for affected in cells:
                chunk_id, _ = self.layout.chunk_for_cell(affected)
                if not history.has_chunk(chunk_id):
                    history.capture(chunk_id, store.load_chunk(session, chunk_id))
            session.set_cells(
                store, {affected: ("uid-a", "a/a.png", 1) for affected in cells}
            )
            history.commit(len(cells))

            # The undo brush covers only part of the painted area.
            subset = cells[: len(cells) // 2]
            restore = {}
            for affected in subset:
                chunk_id, local_index = self.layout.chunk_for_cell(affected)
                previous = history.previous_value(chunk_id, local_index)
                self.assertIsNotNone(previous)
                restore[affected] = previous
            changed = session.set_raw_values(store, restore)
            self.assertEqual(sorted(changed), sorted(subset))

            for affected in subset:
                self.assertEqual(session.get_state(affected, store), (0, 0))
            for affected in cells[len(cells) // 2 :]:
                self.assertNotEqual(session.get_state(affected, store), (0, 0))


class _FakeRecord:
    """Minimal stand-in for BrushRecord: the planner only reads these fields."""

    state = "active"

    def __init__(self, uid, relative_path):
        self.uid = uid
        self.relative_path = relative_path
        self.category_path = relative_path.rsplit("/", 1)[0]


if __name__ == "__main__":
    unittest.main()
