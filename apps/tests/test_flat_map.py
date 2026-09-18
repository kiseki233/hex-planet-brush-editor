from __future__ import annotations

import unittest

from app.flat_map import IcosahedralNetLayout, render_net_surface_ppm
from app.png_pixels import PixelImage
from app.production_topology import ProductionTopology


class FlatMapCoreTests(unittest.TestCase):
    def test_net_contains_all_twenty_faces(self) -> None:
        net = IcosahedralNetLayout(16)
        self.assertEqual(len(net.face_placements), 20)
        self.assertEqual({item.face_id for item in net.face_placements}, set(range(20)))
        left, top, right, bottom = net.bounds
        self.assertLess(left, right)
        self.assertLess(top, bottom)

    def test_face_lattice_points_map_back_to_same_cell(self) -> None:
        topology = ProductionTopology(16)
        net = IcosahedralNetLayout(16)
        for face_id in range(20):
            address = next(
                topology.owner_address(cell_id)
                for a, b, c, cell_id in topology.iter_face_points(face_id)
                if min(a, b, c) > 0 and topology.owner_address(cell_id).face_id == face_id
            )
            point = net.face_point(address)
            hit = net.nearest_cell(topology, *point)
            self.assertIsNotNone(hit)
            self.assertEqual(hit[1], topology.point_id(face_id, *address.weights))

    def test_seam_cells_have_multiple_drawable_copies_but_one_cell_id(self) -> None:
        topology = ProductionTopology(8)
        net = IcosahedralNetLayout(8)
        seam_id = topology.point_id(0, 4, 4, 0)
        representations = topology.cell_representations(seam_id)
        self.assertEqual(len(representations), 2)
        points = [net.face_point(address) for address in representations]
        self.assertNotEqual(points[0], points[1])
        self.assertEqual(
            {net.nearest_cell(topology, *point)[1] for point in points},
            {seam_id},
        )

    def test_visible_cells_are_clipped_to_face_triangles(self) -> None:
        topology = ProductionTopology(8)
        net = IcosahedralNetLayout(8)
        cells = net.visible_cells(topology, net.bounds, maximum_cells=10000)
        self.assertGreater(len(cells), topology.hexagon_count)
        self.assertTrue(all(cell.cell_id >= 12 for cell in cells))
        self.assertTrue(all(len(cell.points) >= 6 for cell in cells))

    def test_flat_surface_renderer_outputs_ppm(self) -> None:
        topology = ProductionTopology(2)
        net = IcosahedralNetLayout(topology.frequency)
        texture = PixelImage(4, 2, 3, bytes((20, 80, 160)) * 8)
        left, top, right, bottom = net.bounds
        ppm = render_net_surface_ppm(
            texture,
            net,
            (left + right) / 2,
            (top + bottom) / 2,
            30.0,
            320,
            180,
            block_size=2,
        )
        self.assertTrue(ppm.startswith(b"P6\n320 180\n255\n"))
        self.assertIn(bytes((20, 80, 160)), ppm)


if __name__ == "__main__":
    unittest.main()
