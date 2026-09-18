from __future__ import annotations

import unittest

from app.chunk_layout import build_chunk_layout
from app.sphere_viewport import SphereViewport, chunk_ids_for_cells
from app.topology import generate_dual_topology


class SphereViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(8)
        cls.layout = build_chunk_layout(cls.topology, target_cells=64)

    def test_projection_returns_front_visible_cells_and_chunks(self) -> None:
        viewport = SphereViewport()
        projection = viewport.project(self.topology, 900, 700, self.layout)
        self.assertGreater(len(projection.visible_cell_ids), 0)
        self.assertLess(len(projection.visible_cell_ids), self.topology.cell_count)
        self.assertEqual(
            projection.visible_chunk_ids,
            chunk_ids_for_cells(self.layout, projection.visible_cell_ids),
        )
        self.assertTrue(all(cell.depth > 0.0 for cell in projection.cells))

    def test_zoom_reduces_visible_cell_count(self) -> None:
        viewport = SphereViewport(zoom=1.0)
        full = viewport.project(self.topology, 900, 700, self.layout)
        viewport.zoom = 4.0
        zoomed = viewport.project(self.topology, 900, 700, self.layout)
        self.assertLess(len(zoomed.visible_cell_ids), len(full.visible_cell_ids))
        self.assertLessEqual(len(zoomed.visible_chunk_ids), len(full.visible_chunk_ids))

    def test_hit_test_uses_projected_polygon(self) -> None:
        viewport = SphereViewport(zoom=2.0)
        projection = viewport.project(self.topology, 900, 700, self.layout)
        target = max(projection.cells, key=lambda cell: cell.depth)
        self.assertEqual(projection.hit_test(target.center_x, target.center_y), target.cell_id)
        self.assertIsNone(projection.hit_test(-1000.0, -1000.0))

    def test_rotation_changes_visible_region(self) -> None:
        viewport = SphereViewport()
        first = viewport.project(self.topology, 900, 700, self.layout)
        viewport.rotate_by(120.0, 0.0)
        second = viewport.project(self.topology, 900, 700, self.layout)
        self.assertNotEqual(set(first.visible_cell_ids), set(second.visible_cell_ids))


if __name__ == "__main__":
    unittest.main()
