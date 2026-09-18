from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapError, SphereMapStore
from app.undo_history import UndoHistory, UndoHistoryError, pack_values, unpack_values


def _paintable_cells(layout, chunk_index, count):
    cell_ids = [cell for cell in layout.chunk_cell_ids(chunk_index) if cell >= 12]
    return cell_ids[:count]


class UndoBufferTest(unittest.TestCase):
    def test_pack_round_trip(self):
        values = [0, 1, 4095, 0xF000, 0xFFFF, 258]
        self.assertEqual(unpack_values(pack_values(values)), values)

    def test_snapshot_is_two_bytes_per_cell(self):
        self.assertEqual(len(pack_values([0] * 258)), 516)


class UndoHistoryStateTest(unittest.TestCase):
    def test_empty_history_reports_nothing_to_do(self):
        history = UndoHistory()
        self.assertFalse(history.can_undo)
        self.assertFalse(history.can_redo)
        self.assertIsNone(history.next_undo_label())

    def test_commit_without_captures_is_dropped(self):
        history = UndoHistory()
        history.begin("空笔划")
        self.assertFalse(history.commit(0))
        self.assertFalse(history.can_undo)

    def test_capture_only_keeps_the_first_snapshot_of_a_chunk(self):
        history = UndoHistory()
        history.begin("绘制")
        history.capture(7, [1, 2, 3])
        history.capture(7, [9, 9, 9])
        self.assertTrue(history.has_chunk(7))
        self.assertEqual(history.previous_value(7, 0), None)
        history.commit(3)
        self.assertEqual(history.previous_value(7, 0), 1)
        self.assertEqual(history.previous_value(7, 2), 3)

    def test_depth_limit_drops_the_oldest_records(self):
        history = UndoHistory(depth=3)
        for index in range(5):
            history.begin(f"第{index}笔")
            history.capture(index, [index])
            history.commit(1)
        self.assertEqual(history.undo_depth, 3)
        self.assertEqual(history.next_undo_label(), "第4笔")

    def test_capturing_blocks_undo(self):
        history = UndoHistory()
        history.begin("绘制")
        history.capture(1, [0])
        with self.assertRaises(UndoHistoryError):
            history.undo(None, None, None)

    def test_previous_value_walks_down_the_stack(self):
        history = UndoHistory()
        history.begin("第一笔")
        history.capture(1, [10, 11])
        history.commit(2)
        history.begin("第二笔")
        history.capture(2, [20, 21])
        history.commit(2)
        # Chunk 1 was only touched by the older stroke; its value still resolves.
        self.assertEqual(history.previous_value(1, 1), 11)
        self.assertEqual(history.previous_value(2, 0), 20)
        self.assertIsNone(history.previous_value(3, 0))


class UndoAgainstMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)

    def _session(self, temporary):
        store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
        return store, store.create_blank("planet", self.layout)

    def test_undo_restores_and_redo_reapplies(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            history = UndoHistory()
            cells = _paintable_cells(self.layout, 0, 4)
            chunk_id, _ = self.layout.chunk_for_cell(cells[0])

            history.begin("绘制 直径3格")
            history.capture(chunk_id, store.load_chunk(session, chunk_id))
            session.set_cells(
                store, {cell: ("uid-city", "城市/a.png", 2) for cell in cells}
            )
            history.commit(len(cells))

            painted = [session.get_state(cell, store) for cell in cells]
            self.assertTrue(all(state[0] != 0 for state in painted))

            result = history.undo(session, store, self.layout)
            self.assertIsNotNone(result)
            self.assertEqual(sorted(result.changed_cell_ids), sorted(cells))
            for cell in cells:
                self.assertEqual(session.get_state(cell, store), (0, 0))
            self.assertIn(chunk_id, session.dirty_chunks)

            redo = history.redo(session, store, self.layout)
            self.assertIsNotNone(redo)
            self.assertEqual(sorted(redo.changed_cell_ids), sorted(cells))
            for cell, state in zip(cells, painted):
                self.assertEqual(session.get_state(cell, store), state)

    def test_undo_does_not_free_brush_table_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            history = UndoHistory()
            cells = _paintable_cells(self.layout, 0, 2)
            chunk_id, _ = self.layout.chunk_for_cell(cells[0])
            history.begin("绘制")
            history.capture(chunk_id, store.load_chunk(session, chunk_id))
            session.set_cells(store, {cells[0]: ("uid-a", "a/a.png", 0)})
            history.commit(1)
            entries_before = dict(session.brush_entries)
            history.undo(session, store, self.layout)
            self.assertEqual(session.brush_entries, entries_before)

    def test_undo_survives_a_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            history = UndoHistory()
            cells = _paintable_cells(self.layout, 1, 3)
            chunk_id, _ = self.layout.chunk_for_cell(cells[0])
            history.begin("绘制")
            history.capture(chunk_id, store.load_chunk(session, chunk_id))
            session.set_cells(
                store, {cell: ("uid-x", "x/x.png", 1) for cell in cells}
            )
            history.commit(len(cells))
            store.save(session)
            self.assertFalse(session.dirty_chunks)

            self.assertTrue(history.can_undo)
            history.undo(session, store, self.layout)
            for cell in cells:
                self.assertEqual(session.get_state(cell, store), (0, 0))
            # Undoing saved work legitimately dirties the map again.
            self.assertTrue(session.dirty_chunks)
            store.save(session)
            reopened = store.open("planet", self.layout)
            for cell in cells:
                self.assertEqual(reopened.get_state(cell, store), (0, 0))

    def test_undo_reports_only_cells_that_actually_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            history = UndoHistory()
            cells = _paintable_cells(self.layout, 0, 3)
            chunk_id, _ = self.layout.chunk_for_cell(cells[0])
            history.begin("绘制")
            history.capture(chunk_id, store.load_chunk(session, chunk_id))
            session.set_cells(store, {cells[0]: ("uid-a", "a/a.png", 0)})
            history.commit(1)
            result = history.undo(session, store, self.layout)
            self.assertEqual(result.changed_cell_ids, (cells[0],))

    def test_set_raw_values_restores_encoded_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            cell = _paintable_cells(self.layout, 0, 1)[0]
            session.set_cells(store, {cell: ("uid-a", "a/a.png", 3)})
            local_id, rotation = session.get_state(cell, store)
            self.assertEqual(rotation, 3)
            changed = session.set_raw_values(store, {cell: 0})
            self.assertEqual(changed, (cell,))
            self.assertEqual(session.get_state(cell, store), (0, 0))
            session.set_raw_values(store, {cell: local_id | (rotation << 12)})
            self.assertEqual(session.get_state(cell, store), (local_id, rotation))

    def test_set_raw_values_rejects_unknown_local_brush_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            cell = _paintable_cells(self.layout, 0, 1)[0]
            with self.assertRaises(SphereMapError):
                session.set_raw_values(store, {cell: 77})

    def test_set_raw_values_skips_reserved_pentagons(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session = self._session(temporary)
            self.assertEqual(session.set_raw_values(store, {0: 0, 5: 0}), ())


if __name__ == "__main__":
    unittest.main()
