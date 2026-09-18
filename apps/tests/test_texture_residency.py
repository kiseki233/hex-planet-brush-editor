from __future__ import annotations

import unittest

from app.texture_residency import TextureResidencyManager


class TextureResidencyTests(unittest.TestCase):
    def test_reference_counts_release_and_reuse_lowest_slot(self) -> None:
        manager = TextureResidencyManager(("empty", "missing"), grace_ticks=0, maximum_layers=8)
        first = manager.allocate("brush-a")
        second = manager.allocate("brush-b")
        self.assertEqual((first.layer, second.layer), (2, 3))
        manager.synchronize((0, 1, 2, 2, 3))
        update = manager.synchronize((0, 1, 3))
        self.assertEqual(update.released_layers, (2,))
        reused = manager.allocate("brush-c")
        self.assertTrue(reused.reused)
        self.assertEqual(reused.layer, 2)
        self.assertEqual(reused.replaced_key, "brush-a")

    def test_grace_ticks_prevent_boundary_churn(self) -> None:
        manager = TextureResidencyManager(("empty", "missing"), grace_ticks=2)
        layer = manager.allocate("brush-a").layer
        manager.synchronize((layer,))
        self.assertEqual(manager.synchronize(()).released_layers, ())
        self.assertEqual(manager.synchronize(()).released_layers, ())
        self.assertEqual(manager.synchronize(()).released_layers, (layer,))

    def test_pinned_slots_never_release(self) -> None:
        manager = TextureResidencyManager(("empty", "missing"), grace_ticks=0)
        update = manager.synchronize(())
        self.assertEqual(update.released_layers, ())
        self.assertEqual(tuple(slot.layer for slot in manager.active_slots()), (0, 1))

    def test_retained_unused_layers_are_lru_evicted_only_under_pressure(self) -> None:
        manager = TextureResidencyManager(
            ("empty", "missing"),
            grace_ticks=0,
            maximum_layers=4,
            retain_unused=True,
        )
        first = manager.allocate("brush-a")
        second = manager.allocate("brush-b")
        manager.synchronize((first.layer,))
        manager.synchronize(())
        self.assertEqual(manager.layer_for_key("brush-a"), first.layer)
        self.assertEqual(manager.layer_for_key("brush-b"), second.layer)
        self.assertEqual(manager.synchronize(()).released_layers, ())

        replacement = manager.allocate("brush-c")
        self.assertTrue(replacement.reused)
        self.assertEqual(replacement.layer, second.layer)
        self.assertEqual(replacement.replaced_key, "brush-b")
        self.assertIsNone(manager.layer_for_key("brush-b"))
        self.assertEqual(manager.layer_for_key("brush-a"), first.layer)


if __name__ == "__main__":
    unittest.main()
