import unittest

from app.zoom_tiers import (
    MAX_ZOOM,
    MIN_ZOOM,
    ZOOM_TIERS,
    ZoomTierPolicy,
    cell_pixels,
    drag_radians_per_pixel,
    tier_ceiling,
    tier_for_zoom,
    visible_cells_estimate,
)


class ZoomTierTableTest(unittest.TestCase):
    def test_six_tiers_cover_the_whole_zoom_range_without_gaps(self):
        self.assertEqual(len(ZOOM_TIERS), 6)
        self.assertEqual(ZOOM_TIERS[-1].zoom_floor, MIN_ZOOM)
        for index, tier in enumerate(ZOOM_TIERS):
            self.assertEqual(tier.index, index)
            if index:
                self.assertEqual(tier_ceiling(tier), ZOOM_TIERS[index - 1].zoom_floor)
                self.assertLess(tier.zoom_floor, ZOOM_TIERS[index - 1].zoom_floor)

    def test_every_tier_is_paintable(self):
        for tier in ZOOM_TIERS:
            self.assertGreaterEqual(tier.min_diameter, 1)
            self.assertLessEqual(tier.max_diameter, 500)
            self.assertLessEqual(tier.min_diameter, tier.suggested_diameter)
            self.assertLessEqual(tier.suggested_diameter, tier.max_diameter)

    def test_far_tiers_require_a_brush_at_least_as_wide_as_their_feedback(self):
        for tier in ZOOM_TIERS:
            if tier.feedback_cells > 1:
                self.assertGreaterEqual(tier.min_diameter, tier.feedback_cells)

    def test_tier_lookup_at_boundaries(self):
        for tier in ZOOM_TIERS:
            self.assertEqual(tier_for_zoom(tier.zoom_floor).index, tier.index)
        self.assertEqual(tier_for_zoom(MAX_ZOOM).index, 0)
        self.assertEqual(tier_for_zoom(MIN_ZOOM).index, 5)
        self.assertEqual(tier_for_zoom(0.01).index, 5)
        self.assertEqual(tier_for_zoom(9999.0).index, 0)

    def test_clamp_diameter(self):
        near = ZOOM_TIERS[0]
        far = ZOOM_TIERS[-1]
        self.assertEqual(near.clamp_diameter(500), near.max_diameter)
        self.assertEqual(far.clamp_diameter(1), far.min_diameter)
        self.assertEqual(far.clamp_diameter(300), 300)


class ZoomTierPolicyTest(unittest.TestCase):
    def test_hysteresis_holds_the_tier_across_a_boundary(self):
        policy = ZoomTierPolicy(initial_zoom=40.0)
        self.assertEqual(policy.tier.index, 1)
        # Just below the L1 floor of 38 but inside the hysteresis band.
        self.assertEqual(policy.update(37.0).index, 1)
        # Far enough below and the tier finally changes.
        self.assertEqual(policy.update(30.0).index, 2)

    def test_policy_is_stable_when_dithering_on_a_boundary(self):
        policy = ZoomTierPolicy(initial_zoom=13.0)
        indexes = {policy.update(zoom).index for zoom in (13.0, 12.9, 13.1, 12.8, 13.2)}
        self.assertEqual(len(indexes), 1)

    def test_policy_clamps_through_the_active_tier(self):
        policy = ZoomTierPolicy(initial_zoom=1.0)
        self.assertEqual(policy.tier.index, 5)
        self.assertEqual(policy.clamp_diameter(1), policy.tier.min_diameter)

    def test_zoom_is_clamped_to_the_supported_range(self):
        policy = ZoomTierPolicy()
        self.assertEqual(policy.update(1e9).index, 0)
        self.assertEqual(policy.update(0.0).index, 5)


class ZoomEstimateTest(unittest.TestCase):
    def test_drag_sensitivity_tracks_inverse_zoom(self):
        self.assertEqual(drag_radians_per_pixel(0.8), drag_radians_per_pixel(1.0))
        self.assertAlmostEqual(
            drag_radians_per_pixel(8.0),
            drag_radians_per_pixel(1.0) / 8.0,
        )
        self.assertAlmostEqual(
            drag_radians_per_pixel(512.0),
            drag_radians_per_pixel(1.0) / 512.0,
        )

    def test_visible_cells_shrink_as_zoom_grows(self):
        counts = [visible_cells_estimate(zoom) for zoom in (1.0, 4.5, 13.0, 38.0, 110.0)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_whole_planet_view_saturates_at_a_hemisphere(self):
        estimate = visible_cells_estimate(0.8)
        self.assertGreater(estimate, 5_000_000)
        self.assertLessEqual(estimate, 5_040_081)

    def test_cell_pixels_grow_with_zoom(self):
        self.assertLess(cell_pixels(1.0), cell_pixels(38.0))
        self.assertGreater(cell_pixels(110.0), 20.0)

    def test_near_tiers_are_at_least_one_pixel_per_cell(self):
        for tier in ZOOM_TIERS:
            if tier.render in {"cell_texture", "cell_color"}:
                self.assertGreaterEqual(cell_pixels(tier.zoom_floor), 1.0)


if __name__ == "__main__":
    unittest.main()
