from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.production_layout import ProductionChunkLayout
from app.production_streaming import ProductionGpuStreamingController
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore


class LodBudgetFallbackTest(unittest.TestCase):
    """A per-cell tier that exceeds the chunk budget degrades, it does not raise.

    The tier ladder is calibrated for a typical window. On a much larger
    viewport the same zoom sees proportionally more chunks, and this used to
    raise ProductionStreamingError - which only reached a status line, leaving
    the viewport frozen on stale content.
    """

    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(64)
        cls.layout = ProductionChunkLayout(cls.topology)
        cls.visibility = build_production_visibility_index(cls.layout)

    def _controller(self, temporary, maximum_visible_chunks):
        store = SphereMapStore(Path(temporary) / "maps")
        session = store.create_blank("planet", self.layout)
        return ProductionGpuStreamingController(
            self.topology, self.layout, self.visibility, session, store, {},
            Path(temporary) / "brushes",
            lod_level=4, automatic_lod=True,
            maximum_visible_chunks=maximum_visible_chunks,
        )

    def test_exceeding_the_budget_falls_back_to_the_far_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(temporary, maximum_visible_chunks=4)
            # A detail tier zoom whose chunk count is far above the tiny budget.
            frame = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            self.assertEqual(frame.lod.level, 4)
            self.assertEqual(frame.lod.mode, "aggregate")
            self.assertIn("超出", frame.lod.description)
            self.assertIsNotNone(frame.reset_batch)
            self.assertEqual(frame.reset_batch.instance_count, 0)

    def test_fallback_still_produces_an_editable_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(temporary, maximum_visible_chunks=4)
            frame = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            patch = controller.patch_for_frame(1, frame)
            self.assertTrue(patch.editable)

    def test_a_generous_budget_keeps_the_detail_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(temporary, maximum_visible_chunks=100_000)
            frame = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            self.assertLess(frame.lod.level, 4)

    def test_recovery_after_a_fallback_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(temporary, maximum_visible_chunks=4)
            controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            # Zooming further in shrinks the chunk count under the budget again.
            frame = controller.update_view(0.0, 0.0, 400.0, 1100, 760)
            self.assertLess(frame.lod.level, 4)


if __name__ == "__main__":
    unittest.main()
