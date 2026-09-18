from __future__ import annotations

import unittest

from app.brush_lod import LOD_EFFECTIVE_SIZES
from app.production_lod import ProductionLodController
from app.zoom_tiers import ZOOM_TIERS


class ProductionLodTests(unittest.TestCase):
    """The GPU level is selected by the same tier table the interface shows."""

    def test_zoom_ladder_walks_lod4_down_to_lod0(self) -> None:
        controller = ProductionLodController(initial_level=4)
        observed = [controller.update(zoom).level for zoom in (1.0, 5.0, 20.0, 50.0, 150.0)]
        self.assertEqual(observed, [4, 3, 2, 1, 0])

    def test_every_tier_maps_to_its_declared_texture_level(self) -> None:
        for tier in ZOOM_TIERS:
            controller = ProductionLodController(initial_level=4)
            # Step in from the far view so hysteresis cannot mask a wrong mapping.
            controller.update(0.8)
            decision = controller.update(tier.zoom_floor * 1.2)
            expected = 4 if tier.texture_lod is None else tier.texture_lod
            self.assertEqual(
                decision.level, expected, f"{tier.name} at zoom {tier.zoom_floor * 1.2}"
            )

    def test_hysteresis_holds_the_level_across_a_boundary(self) -> None:
        controller = ProductionLodController(initial_level=4)
        self.assertEqual(controller.update(40.0).level, 1)
        # Just under the L1 floor of 38 but inside the hysteresis band.
        self.assertEqual(controller.update(37.0).level, 1)
        self.assertEqual(controller.update(30.0).level, 2)

    def test_changed_flag_only_fires_on_a_real_transition(self) -> None:
        controller = ProductionLodController(initial_level=4)
        self.assertTrue(controller.update(150.0).changed)
        self.assertFalse(controller.update(150.0).changed)
        self.assertFalse(controller.update(160.0).changed)
        self.assertTrue(controller.update(1.0).changed)

    def test_decision_carries_the_tier_and_a_matching_description(self) -> None:
        controller = ProductionLodController(initial_level=4)
        decision = controller.update(150.0)
        self.assertIsNotNone(decision.tier)
        self.assertEqual(decision.tier.name, "L0 原图")
        self.assertIn(decision.tier.name, decision.description)
        self.assertIn(str(LOD_EFFECTIVE_SIZES[0]), decision.description)
        self.assertTrue(decision.detailed)

    def test_far_view_reports_aggregate_mode(self) -> None:
        controller = ProductionLodController(initial_level=4)
        decision = controller.update(1.0)
        self.assertEqual(decision.level, 4)
        self.assertEqual(decision.mode, "aggregate")
        self.assertFalse(decision.detailed)

    def test_candidate_cells_shrink_as_the_view_closes_in(self) -> None:
        controller = ProductionLodController(initial_level=4)
        counts = [controller.update(zoom).candidate_cells for zoom in (1.0, 5.0, 20.0, 150.0)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_initial_level_seeds_the_tier_without_a_spurious_change(self) -> None:
        controller = ProductionLodController(initial_level=0)
        self.assertEqual(controller.level, 0)
        self.assertFalse(controller.update(150.0).changed)


if __name__ == "__main__":
    unittest.main()
