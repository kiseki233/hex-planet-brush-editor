from __future__ import annotations

import math
import unittest

from app.gpu_batch import INSTANCE_STRUCT, GpuInstance, _exact_cell_corners
from app.gpu_native import (
    SPHERE_FRAGMENT_SHADER_SOURCE,
    SPHERE_VERTEX_SHADER_SOURCE,
    VERTEX_SHADER_SOURCE,
    _pixel_image_rgba,
)
from app.png_pixels import PixelImage
from app.production_topology import ProductionTopology
from app.software_globe import (
    average_edge_pixels,
    draw_shaded_sphere,
    point_in_polygon,
    project_cell_polygon,
    render_textured_globe_view_ppm,
    render_textured_sphere_ppm,
)


class _FakeCanvas:
    def __init__(self) -> None:
        self.ovals = []
        self.polygons = []

    def create_oval(self, *args, **kwargs):
        self.ovals.append((args, kwargs))
        return len(self.ovals)

    def create_polygon(self, *args, **kwargs):
        self.polygons.append((args, kwargs))
        return len(self.polygons)


class SoftwareGlobeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = ProductionTopology(16)

    def test_far_sphere_is_continuous_oval_layers_not_fake_hexagons(self) -> None:
        canvas = _FakeCanvas()
        draw_shaded_sphere(canvas, 400.0, 300.0, 240.0)
        self.assertGreaterEqual(len(canvas.ovals), 40)
        self.assertEqual(canvas.polygons, [])
        outer = canvas.ovals[0][0]
        self.assertEqual(outer, (160.0, 60.0, 640.0, 540.0))

    def test_adjacent_projected_cells_share_exact_screen_edge(self) -> None:
        first = second = None
        cell_id = neighbor = -1
        for candidate in range(12, self.topology.cell_count):
            projected_candidate = project_cell_polygon(
                self.topology, candidate, 0.0, 0.0, 500.0, 500.0, 8000.0, 1000, 1000,
                margin=10000.0,
            )
            if projected_candidate is None:
                continue
            for candidate_neighbor in self.topology.cell_neighbors(candidate):
                if candidate_neighbor < 12:
                    continue
                projected_neighbor = project_cell_polygon(
                    self.topology, candidate_neighbor, 0.0, 0.0, 500.0, 500.0, 8000.0, 1000, 1000,
                    margin=10000.0,
                )
                if projected_neighbor is not None:
                    cell_id, neighbor = candidate, candidate_neighbor
                    first, second = projected_candidate, projected_neighbor
                    break
            if first is not None and second is not None:
                break
        self.assertGreaterEqual(cell_id, 12)
        self.assertGreaterEqual(neighbor, 12)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        first_points = {(round(first[0][i], 8), round(first[0][i + 1], 8)) for i in range(0, 12, 2)}
        second_points = {(round(second[0][i], 8), round(second[0][i + 1], 8)) for i in range(0, 12, 2)}
        self.assertEqual(len(first_points & second_points), 2)
        self.assertGreater(average_edge_pixels(first[0]), 1.0)

    def test_gpu_instance_packs_six_exact_corners(self) -> None:
        cell_id = 12
        corners = _exact_cell_corners(self.topology, cell_id)
        instance = GpuInstance(cell_id, corners, 3, 5)
        self.assertEqual(len(instance.packed()), INSTANCE_STRUCT.size)
        self.assertEqual(INSTANCE_STRUCT.size, 80)
        self.assertEqual(instance.corners, self.topology.cell_corners(cell_id))

    def test_shader_uses_exact_corner_attributes_and_sphere_underlay(self) -> None:
        self.assertIn("inCorner0", VERTEX_SHADER_SOURCE)
        self.assertIn("inCorner5", VERTEX_SHADER_SOURCE)
        self.assertNotIn("inRadius", VERTEX_SHADER_SOURCE)
        self.assertIn("inPosition", SPHERE_VERTEX_SHADER_SOURCE)
        self.assertIn("lightDirection", SPHERE_FRAGMENT_SHADER_SOURCE)
        self.assertIn("sampler2D uSurface", SPHERE_FRAGMENT_SHADER_SOURCE)
        self.assertIn("texture(uSurface", SPHERE_FRAGMENT_SHADER_SOURCE)

    def test_gpu_surface_texture_expands_rgb_to_rgba(self) -> None:
        image = PixelImage(2, 1, 3, bytes((1, 2, 3, 4, 5, 6)))
        self.assertEqual(
            _pixel_image_rgba(image),
            bytes((1, 2, 3, 255, 4, 5, 6, 255)),
        )


    def test_textured_globe_rotates_equirectangular_surface(self) -> None:
        width, height = 8, 4
        pixels = bytearray()
        for _y in range(height):
            for x in range(width):
                pixels.extend((30, 40, 220) if x >= 5 else (220, 40, 30))
        texture = PixelImage(width, height, 3, bytes(pixels))
        first = render_textured_globe_view_ppm(
            texture, 0.0, 0.0, 40, 40, 20.0, 20.0, 18.0, block_size=1
        )
        second = render_textured_globe_view_ppm(
            texture, -math.pi / 2.0, 0.0, 40, 40, 20.0, 20.0, 18.0, block_size=1
        )

        def center_rgb(payload: bytes) -> tuple[int, int, int]:
            header_end = payload.find(b"\n255\n") + len(b"\n255\n")
            raw = payload[header_end:]
            offset = (20 * 40 + 20) * 3
            return tuple(raw[offset : offset + 3])

        first_color = center_rgb(first)
        second_color = center_rgb(second)
        self.assertGreater(first_color[2], first_color[0])
        self.assertGreater(second_color[0], second_color[2])

    def test_mini_globe_renderer_returns_square_and_tracks_orientation(self) -> None:
        width, height = 8, 4
        pixels = bytearray()
        for _y in range(height):
            for x in range(width):
                pixels.extend((30, 40, 220) if x >= 5 else (220, 40, 30))
        texture = PixelImage(width, height, 3, bytes(pixels))
        first = render_textured_sphere_ppm(
            texture, 0.0, 0.0, 176, block_size=2
        )
        second = render_textured_sphere_ppm(
            texture, -math.pi / 2.0, 0.0, 176, block_size=2
        )

        header = b"P6\n176 176\n255\n"
        self.assertTrue(first.startswith(header))
        self.assertEqual(len(first) - len(header), 176 * 176 * 3)
        offset = len(header) + (88 * 176 + 88) * 3
        first_color = tuple(first[offset : offset + 3])
        second_color = tuple(second[offset : offset + 3])
        self.assertGreater(first_color[2], first_color[0])
        self.assertGreater(second_color[0], second_color[2])

    def test_point_in_polygon(self) -> None:
        square = (0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0)
        self.assertTrue(point_in_polygon(5.0, 5.0, square))
        self.assertFalse(point_in_polygon(12.0, 5.0, square))


if __name__ == "__main__":
    unittest.main()
