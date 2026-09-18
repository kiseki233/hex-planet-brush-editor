import math
import random
import unittest

from app.production_topology import ProductionTopology
from app.software_globe import rotate_point
from app.sphere_picker import (
    SphereScreenPicker,
    cap_boundary_directions,
    cell_angular_pitch,
    project_direction,
    rotate_to_view,
    screen_to_direction,
)


class SpherePickerTest(unittest.TestCase):
    def test_rotation_matches_the_software_renderer(self):
        rng = random.Random(7)
        for _ in range(500):
            point = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            yaw = rng.uniform(-3.2, 3.2)
            pitch = rng.uniform(-1.45, 1.45)
            mine = rotate_to_view(point, yaw, pitch)
            theirs = rotate_point(point, yaw, pitch)
            for left, right in zip(mine, theirs):
                self.assertAlmostEqual(left, right, places=12)

    def test_projection_round_trip(self):
        rng = random.Random(11)
        checked = 0
        for _ in range(500):
            yaw = rng.uniform(-3.2, 3.2)
            pitch = rng.uniform(-1.45, 1.45)
            radius = rng.uniform(50.0, 20000.0)
            direction = _random_unit(rng)
            screen_x, screen_y, depth = project_direction(
                direction, 640.0, 410.0, radius, yaw, pitch
            )
            if depth <= 1e-4:
                continue
            back = screen_to_direction(
                screen_x, screen_y, 640.0, 410.0, radius, yaw, pitch
            )
            self.assertIsNotNone(back)
            for left, right in zip(direction, back):
                self.assertAlmostEqual(left, right, places=9)
            checked += 1
        self.assertGreater(checked, 100)

    def test_outside_the_disc_returns_none(self):
        self.assertIsNone(screen_to_direction(0.0, 0.0, 500.0, 500.0, 10.0, 0.0, 0.0))

    def test_every_cell_center_resolves_to_its_own_cell(self):
        for frequency in (12, 64):
            topology = ProductionTopology(frequency)
            picker = SphereScreenPicker(topology)
            for cell_id in range(topology.cell_count):
                self.assertEqual(
                    picker.cell_at_direction(topology.cell_center(cell_id)), cell_id
                )

    def test_production_frequency_cell_centers_resolve(self):
        topology = ProductionTopology(1004)
        picker = SphereScreenPicker(topology)
        rng = random.Random(23)
        for _ in range(400):
            cell_id = rng.randrange(topology.cell_count)
            self.assertEqual(
                picker.cell_at_direction(topology.cell_center(cell_id)), cell_id
            )

    def test_picked_cell_is_the_true_nearest_center(self):
        topology = ProductionTopology(16)
        picker = SphereScreenPicker(topology)
        rng = random.Random(31)
        centers = [topology.cell_center(cell_id) for cell_id in range(topology.cell_count)]
        for _ in range(200):
            direction = _random_unit(rng)
            picked = picker.cell_at_direction(direction)
            picked_dot = _dot(centers[picked], direction)
            best_dot = max(_dot(center, direction) for center in centers)
            self.assertAlmostEqual(picked_dot, best_dot, places=12)

    def test_pick_works_at_every_zoom_level(self):
        topology = ProductionTopology(1004)
        picker = SphereScreenPicker(topology)
        width = height = 900
        center_x = center_y = 450.0
        for zoom in (0.8, 1.0, 4.5, 13.0, 38.0, 110.0, 512.0):
            radius = min(width, height) * 0.42 * zoom
            cell_id = picker.pick(
                center_x + 3.0, center_y - 5.0, center_x, center_y, radius, -0.35, 0.25
            )
            self.assertIsNotNone(cell_id, f"zoom {zoom} produced no pick")
            self.assertGreaterEqual(cell_id, 0)
            self.assertLess(cell_id, topology.cell_count)

    def test_pick_outside_the_planet_is_none(self):
        topology = ProductionTopology(64)
        picker = SphereScreenPicker(topology)
        self.assertIsNone(picker.pick(0.0, 0.0, 400.0, 400.0, 100.0, 0.0, 0.0))

    def test_cap_boundary_stays_at_the_requested_angle(self):
        direction = (0.3, 0.5, 0.81)
        length = math.sqrt(sum(value * value for value in direction))
        unit = tuple(value / length for value in direction)
        angle = 0.2
        for sample in cap_boundary_directions(unit, angle, samples=24):
            self.assertAlmostEqual(math.acos(max(-1.0, min(1.0, _dot(sample, unit)))), angle, places=9)

    def test_cell_angular_pitch_matches_the_grid(self):
        topology = ProductionTopology(1004)
        pitch = cell_angular_pitch(topology.frequency)
        cell_id = 500_000
        neighbors = topology.cell_neighbor_ids_unordered(cell_id)
        center = topology.cell_center(cell_id)
        for neighbor_id in neighbors:
            angle = math.acos(
                max(-1.0, min(1.0, _dot(center, topology.cell_center(neighbor_id))))
            )
            self.assertLess(abs(angle - pitch) / pitch, 0.35)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _random_unit(rng):
    while True:
        candidate = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        length = math.sqrt(sum(value * value for value in candidate))
        if 0.2 < length <= 1.0:
            return tuple(value / length for value in candidate)


if __name__ == "__main__":
    unittest.main()
