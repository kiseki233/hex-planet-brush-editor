from __future__ import annotations

import unittest

from app.stream_scheduler import VisibleChunkScheduler


class StreamSchedulerTests(unittest.TestCase):
    def test_budgeted_transition_is_dense_and_completes(self) -> None:
        scheduler = VisibleChunkScheduler(max_additions=2, max_removals=1)
        active = {1, 2, 3}
        desired = {3, 4, 5, 6}
        first = scheduler.schedule(active, desired, {4: 0.2, 5: 0.1, 6: 0.3, 1: 9.0, 2: 8.0})
        self.assertEqual(first.remove_chunk_ids, (1,))
        self.assertEqual(first.add_chunk_ids, (5, 4))
        self.assertFalse(first.complete)
        second = scheduler.schedule(first.next_active_chunk_ids, desired)
        self.assertEqual(second.remove_chunk_ids, (2,))
        self.assertEqual(second.add_chunk_ids, (6,))
        self.assertTrue(second.complete)
        self.assertEqual(set(second.next_active_chunk_ids), desired)


if __name__ == "__main__":
    unittest.main()
