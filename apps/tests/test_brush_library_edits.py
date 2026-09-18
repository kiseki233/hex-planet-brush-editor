from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.brush_stroke import records_for_group
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from tests.test_helpers import write_rgb_png


class BrushLibraryEditTest(unittest.TestCase):
    """Adding, deleting and moving brush files, and what the map sees.

    Brush identity is what protects painted cells: the map stores uids, so a
    refresh that mints a new uid for an existing image silently turns every cell
    painted with it into missing texture.
    """

    def _library(self, temporary, files):
        brush_root = Path(temporary) / "brushes"
        for relative, rgb in files.items():
            write_rgb_png(brush_root / relative, rgb=rgb)
        catalog = BrushCatalog(brush_root)
        records = {r.uid: r for r in catalog.scan().records}
        return brush_root, catalog, records

    def _painted_map(self, temporary, records, names):
        topology = ProductionTopology(32)
        layout = ProductionChunkLayout(topology)
        store = SphereMapStore(Path(temporary) / "maps")
        session = store.create_blank("planet", layout)
        cells = [c for c in layout.chunks[0].cell_ids if c >= 12][: len(names) * 2]
        by_name = {Path(r.relative_path).name: r for r in records.values()}
        for index, cell in enumerate(cells):
            record = by_name[names[index % len(names)]]
            session.set_cells(store, {cell: (record.uid, record.relative_path, 0)})
        store.save(session)
        return store, session, cells

    def _unresolved(self, records, session, store, cells):
        count = 0
        for cell in cells:
            local_id, _ = session.get_state(cell, store)
            if local_id == 0:
                continue
            record = records.get(session.brush_entries[local_id].brush_uid)
            if record is None or record.state != "active":
                count += 1
        return count

    # -- adding ---------------------------------------------------------

    def test_added_brush_joins_the_group_without_touching_painted_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60)}
            )
            store, session, cells = self._painted_map(temporary, records, ["a.png"])
            before = {r.uid for r in records.values()}

            write_rgb_png(brush_root / "森林" / "b.png", rgb=(40, 170, 80))
            rescanned = {r.uid: r for r in catalog.scan().records}

            self.assertTrue(before.issubset(rescanned))
            self.assertEqual(len(rescanned), 2)
            self.assertEqual(len(records_for_group(rescanned, "森林")), 2)
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    def test_added_brush_does_not_consume_a_map_brush_table_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60)}
            )
            _store, session, _cells = self._painted_map(temporary, records, ["a.png"])
            used = len(session.brush_entries)
            write_rgb_png(brush_root / "森林" / "b.png", rgb=(40, 170, 80))
            catalog.scan()
            # The table only grows when a brush is actually painted with.
            self.assertEqual(len(session.brush_entries), used)

    # -- deleting -------------------------------------------------------

    def test_deleted_brush_is_retired_but_its_cells_keep_their_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60), "森林/b.png": (40, 170, 80)}
            )
            store, session, cells = self._painted_map(
                temporary, records, ["a.png", "b.png"]
            )
            gone = next(
                r for r in records.values() if r.relative_path.endswith("a.png")
            )
            (brush_root / "森林" / "a.png").unlink()
            rescanned = {r.uid: r for r in catalog.scan().records}

            self.assertEqual(rescanned[gone.uid].state, "missing")
            self.assertEqual(len(records_for_group(rescanned, "森林")), 1)
            self.assertEqual(
                self._unresolved(rescanned, session, store, cells), len(cells) // 2
            )
            # The map still names the brush, so nothing is lost on disk.
            self.assertIn(
                gone.uid, {e.brush_uid for e in session.brush_entries.values()}
            )

    def test_restoring_a_deleted_file_reactivates_the_same_uid(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60)}
            )
            store, session, cells = self._painted_map(temporary, records, ["a.png"])
            uid = next(iter(records))
            (brush_root / "森林" / "a.png").unlink()
            catalog.scan()
            write_rgb_png(brush_root / "森林" / "a.png", rgb=(20, 150, 60))
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertEqual(rescanned[uid].state, "active")
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    # -- moving ---------------------------------------------------------

    def test_moving_a_brush_keeps_its_uid_and_changes_its_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60)}
            )
            store, session, cells = self._painted_map(temporary, records, ["a.png"])
            uid = next(iter(records))
            (brush_root / "沙漠").mkdir(parents=True, exist_ok=True)
            shutil.move(brush_root / "森林" / "a.png", brush_root / "沙漠" / "a.png")
            rescanned = {r.uid: r for r in catalog.scan().records}

            self.assertEqual(rescanned[uid].state, "active")
            self.assertEqual(rescanned[uid].category_path, "沙漠")
            self.assertEqual(len(records_for_group(rescanned, "森林")), 0)
            self.assertEqual(len(records_for_group(rescanned, "沙漠")), 1)
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    def test_renaming_a_brush_keeps_its_uid(self):
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, catalog, records = self._library(
                temporary, {"森林/a.png": (20, 150, 60)}
            )
            store, session, cells = self._painted_map(temporary, records, ["a.png"])
            uid = next(iter(records))
            (brush_root / "森林" / "a.png").rename(brush_root / "森林" / "renamed.png")
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertEqual(rescanned[uid].state, "active")
            self.assertTrue(rescanned[uid].relative_path.endswith("renamed.png"))
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    def test_moving_several_identical_brushes_together_keeps_every_uid(self):
        """The regression: identical content used to defeat move detection."""
        with tempfile.TemporaryDirectory() as temporary:
            files = {f"旧/same{i}.png": (99, 99, 99) for i in range(1, 4)}
            brush_root, catalog, records = self._library(temporary, files)
            store, session, cells = self._painted_map(
                temporary, records, ["same1.png", "same2.png", "same3.png"]
            )
            before = {r.uid for r in records.values()}
            self.assertEqual(len(before), 3)

            (brush_root / "新").mkdir(parents=True, exist_ok=True)
            for index in range(1, 4):
                shutil.move(
                    brush_root / "旧" / f"same{index}.png",
                    brush_root / "新" / f"same{index}.png",
                )
            rescanned = {r.uid: r for r in catalog.scan().records}

            self.assertEqual(len(rescanned), 3)
            self.assertEqual(before, set(rescanned))
            for record in rescanned.values():
                self.assertEqual(record.state, "active")
                self.assertEqual(record.category_path, "新")
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    def test_moving_a_whole_folder_of_distinct_brushes_keeps_every_uid(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = {f"旧/b{i}.png": (10 * i, 120, 60) for i in range(1, 6)}
            brush_root, catalog, records = self._library(temporary, files)
            store, session, cells = self._painted_map(
                temporary, records, [f"b{i}.png" for i in range(1, 6)]
            )
            before = {r.uid for r in records.values()}
            shutil.move(brush_root / "旧", brush_root / "新")
            rescanned = {r.uid: r for r in catalog.scan().records}
            self.assertEqual(before, set(rescanned))
            self.assertEqual(self._unresolved(rescanned, session, store, cells), 0)

    def test_a_partial_move_of_identical_brushes_retires_only_the_leftovers(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = {f"旧/same{i}.png": (99, 99, 99) for i in range(1, 4)}
            brush_root, catalog, records = self._library(temporary, files)
            (brush_root / "新").mkdir(parents=True, exist_ok=True)
            # Move two of the three, and delete the third outright.
            shutil.move(brush_root / "旧" / "same1.png", brush_root / "新" / "same1.png")
            shutil.move(brush_root / "旧" / "same2.png", brush_root / "新" / "same2.png")
            (brush_root / "旧" / "same3.png").unlink()
            rescanned = catalog.scan()
            self.assertEqual(rescanned.active_count, 2)
            self.assertEqual(rescanned.missing_count, 1)
            self.assertEqual(len(rescanned.records), 3)


if __name__ == "__main__":
    unittest.main()
