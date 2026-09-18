from __future__ import annotations

import unittest

from app.gpu_memory import (
    DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
    budgeted_texture_layer_limit,
    catalog_texture_layer_limit,
    next_texture_capacity,
    texture_layer_bytes,
)


class GpuMemoryBudgetTests(unittest.TestCase):
    def test_lod0_limit_stays_within_three_gibibytes(self) -> None:
        limit = budgeted_texture_layer_limit(520)
        self.assertEqual(limit, DEFAULT_GPU_TEXTURE_BUDGET_BYTES // texture_layer_bytes(520))
        self.assertLessEqual(
            limit * texture_layer_bytes(520),
            DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
        )
        self.assertGreaterEqual(limit, 2242)

    def test_catalog_limit_keeps_headroom_without_requesting_empty_layers_forever(self) -> None:
        limit = catalog_texture_layer_limit(
            2240,
            padded_size=520,
            headroom=512,
        )
        self.assertEqual(limit, 2754)
        self.assertLess(
            limit * texture_layer_bytes(520),
            DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
        )
        self.assertEqual(
            catalog_texture_layer_limit(0, padded_size=520),
            1024,
        )

    def test_growth_clamps_to_budget_instead_of_next_power_of_two(self) -> None:
        maximum = 2754
        self.assertEqual(next_texture_capacity(1025, 1024, maximum), 2048)
        self.assertEqual(next_texture_capacity(2049, 2048, maximum), maximum)


if __name__ == "__main__":
    unittest.main()
