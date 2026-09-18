from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.brush_stroke import (
    BrushStrokePlanner,
    BrushStrokeState,
    BrushStrokeTool,
    random_assignments,
    records_for_group,
)
from app.chunk_layout import build_chunk_layout
from app.production_topology import ProductionTopology
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology
from tests.test_helpers import write_rgb_png


class BrushStrokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = ProductionTopology(16)
        cls.planner = BrushStrokePlanner(cls.topology)

    def test_continuous_path_has_adjacent_steps_and_no_gaps(self) -> None:
        start = 120
        first = self.topology.cell_neighbor_ids_unordered(start)[0]
        end = self.topology.cell_neighbor_ids_unordered(first)[2]
        path = self.planner.connect(start, end)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        for left, right in zip(path, path[1:]):
            self.assertIn(right, self.topology.cell_neighbor_ids_unordered(left))

    def test_same_stroke_does_not_repaint_the_same_cells(self) -> None:
        state = BrushStrokeState(BrushStrokeTool("erase", 5))
        center = 200
        first = self.planner.plan_segment(state, center)
        second = self.planner.plan_segment(state, center)
        self.assertGreater(len(first.affected_cell_ids), 1)
        self.assertEqual(second.affected_cell_ids, ())
        new_stroke = BrushStrokeState(BrushStrokeTool("erase", 5))
        third = self.planner.plan_segment(new_stroke, center)
        self.assertEqual(set(first.affected_cell_ids), set(third.affected_cell_ids))

    def test_diameter_range_and_large_disk_growth(self) -> None:
        center = 300
        one = self.planner.plan_segment(
            BrushStrokeState(BrushStrokeTool("erase", 1)), center
        )
        ten = self.planner.plan_segment(
            BrushStrokeState(BrushStrokeTool("erase", 10)), center
        )
        self.assertEqual(len(one.affected_cell_ids), 1)
        self.assertGreater(len(ten.affected_cell_ids), len(one.affected_cell_ids))
        self.assertEqual(BrushStrokePlanner.graph_radius(500), 250)

    def test_group_selection_includes_descendant_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            write_rgb_png(root / "城市" / "a.png", rgb=(1, 2, 3))
            write_rgb_png(root / "城市" / "现代" / "b.png", rgb=(4, 5, 6))
            write_rgb_png(root / "海洋" / "c.png", rgb=(7, 8, 9))
            scan = BrushCatalog(root).scan()
            records = {item.uid: item for item in scan.records}
            city = records_for_group(records, "城市")
            self.assertEqual(len(city), 2)
            self.assertTrue(all(item.category_path.startswith("城市") for item in city))

    def test_random_group_assignment_uses_images_and_six_rotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            write_rgb_png(root / "城市" / "a.png", rgb=(1, 2, 3))
            write_rgb_png(root / "城市" / "b.png", rgb=(4, 5, 6))
            scan = BrushCatalog(root).scan()
            records = tuple(scan.active_records)
            assignments = random_assignments(
                range(1000, 1200),
                BrushStrokeTool("paint", 1, records),
                rng=random.Random(20260726),
            )
            used_uids = {value[0] for value in assignments.values() if value is not None}
            rotations = {value[2] for value in assignments.values() if value is not None}
            self.assertEqual(used_uids, {record.uid for record in records})
            self.assertEqual(rotations, set(range(6)))

    def test_batch_write_completely_overwrites_previous_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = generate_dual_topology(4)
            layout = build_chunk_layout(topology, target_cells=32)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("planet", layout)
            assignments = {
                12: ("uid-a", "城市/a.png", 1),
                13: ("uid-b", "城市/b.png", 5),
            }
            changed = session.set_cells(store, assignments)
            self.assertEqual(set(changed), {12, 13})
            overwrite = session.set_cells(
                store,
                {
                    12: ("uid-b", "城市/b.png", 3),
                    13: None,
                },
            )
            self.assertEqual(set(overwrite), {12, 13})
            store.save(session)
            reopened = store.open("planet", layout)
            self.assertEqual(reopened.brush_uid_for_cell(12, store), ("uid-b", 3))
            self.assertEqual(reopened.brush_uid_for_cell(13, store), (None, 0))


if __name__ == "__main__":
    unittest.main()
