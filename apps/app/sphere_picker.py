from __future__ import annotations

import math
from typing import Sequence

from .topology import BASE_FACES, BASE_VERTICES

# Adjacent dual-cell centers on a frequency-N icosahedral grid are separated by
# close to 1.2 / N radians.  ``brush_stroke`` already relies on this constant for
# its A* heuristic; the brush cursor uses it to convert a cell radius into an
# angular radius.
CELL_ANGULAR_PITCH_NUMERATOR = 1.2


class SpherePickError(ValueError):
    pass


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _normalize(value: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(value, value))
    if length <= 1e-15:
        raise SpherePickError("Cannot normalize a zero-length direction")
    return (float(value[0]) / length, float(value[1]) / length, float(value[2]) / length)


def cell_angular_pitch(frequency: int) -> float:
    return CELL_ANGULAR_PITCH_NUMERATOR / max(1, int(frequency))


def rotate_to_view(
    point: Sequence[float], yaw: float, pitch: float
) -> tuple[float, float, float]:
    """Match ``software_globe.rotate_point`` exactly.

    The picker inverts this transform, so any divergence would offset every
    click.  It is duplicated here rather than imported so that the inverse can be
    unit-tested against the renderer without a circular import.
    """
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    x, z = x * cosine_yaw + z * sine_yaw, -x * sine_yaw + z * cosine_yaw
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    return x, y * cosine_pitch - z * sine_pitch, y * sine_pitch + z * cosine_pitch


def rotate_from_view(
    point: Sequence[float], yaw: float, pitch: float
) -> tuple[float, float, float]:
    """Inverse of :func:`rotate_to_view`."""
    x_view, y_view, z_view = float(point[0]), float(point[1]), float(point[2])
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    y_world = y_view * cosine_pitch + z_view * sine_pitch
    z_yawed = -y_view * sine_pitch + z_view * cosine_pitch
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    x_world = x_view * cosine_yaw - z_yawed * sine_yaw
    z_world = x_view * sine_yaw + z_yawed * cosine_yaw
    return x_world, y_world, z_world


def screen_to_direction(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    radius: float,
    yaw: float,
    pitch: float,
) -> tuple[float, float, float] | None:
    """Invert the orthographic sphere projection used by the Tk viewport.

    Returns ``None`` when the point falls outside the projected planet disc.
    ``radius`` is passed in rather than recomputed so that the picker always
    matches whatever radius the caller actually drew with.
    """
    if radius <= 0.0:
        return None
    view_x = (float(x) - float(center_x)) / radius
    view_y = (float(center_y) - float(y)) / radius
    radius_squared = view_x * view_x + view_y * view_y
    if radius_squared > 1.0:
        return None
    view_z = math.sqrt(max(0.0, 1.0 - radius_squared))
    return rotate_from_view((view_x, view_y, view_z), yaw, pitch)


def project_direction(
    direction: Sequence[float],
    center_x: float,
    center_y: float,
    radius: float,
    yaw: float,
    pitch: float,
) -> tuple[float, float, float]:
    """Project a unit direction to canvas coordinates plus its view depth."""
    x, y, z = rotate_to_view(direction, yaw, pitch)
    return center_x + x * radius, center_y - y * radius, z


def cap_boundary_directions(
    direction: Sequence[float], angular_radius: float, samples: int = 36
) -> tuple[tuple[float, float, float], ...]:
    """Sample the boundary of a spherical cap around ``direction``."""
    samples = max(3, int(samples))
    center = _normalize(direction)
    # Any vector not parallel to the center works as a basis seed.
    seed = (0.0, 0.0, 1.0) if abs(center[2]) < 0.9 else (1.0, 0.0, 0.0)
    tangent_u = _normalize(_cross(center, seed))
    tangent_v = _cross(center, tangent_u)
    cosine = math.cos(angular_radius)
    sine = math.sin(angular_radius)
    result: list[tuple[float, float, float]] = []
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        offset_x = math.cos(angle) * sine
        offset_y = math.sin(angle) * sine
        result.append(
            (
                center[0] * cosine + tangent_u[0] * offset_x + tangent_v[0] * offset_y,
                center[1] * cosine + tangent_u[1] * offset_x + tangent_v[1] * offset_y,
                center[2] * cosine + tangent_u[2] * offset_x + tangent_v[2] * offset_y,
            )
        )
    return tuple(result)


class SphereScreenPicker:
    """Resolve a screen position to an exact CellId at any zoom level.

    The lattice point is found analytically: a direction lies in exactly one base
    face, its barycentric weights against that face's three vertices are a linear
    solve, and rounding those weights to the frequency grid names a CellId
    directly through ``topology.point_id``.  Because ``cell_center`` normalizes a
    barycentric combination, that inverse is exact in the face interior but can
    land one cell off near a base edge, where the gnomonic distortion is largest.
    A short bounded hill climb over the neighbour graph removes that error, so the
    result is always the true nearest cell center.

    Nothing here depends on what is currently rendered, which is what makes every
    zoom tier editable.
    """

    REFINE_STEPS = 8

    def __init__(self, topology) -> None:
        self.topology = topology
        self.frequency = int(topology.frequency)
        self._face_inverses = self._build_face_inverses()
        self.last_cell_id: int | None = None

    @staticmethod
    def _build_face_inverses() -> tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        ...,
    ]:
        inverses = []
        for face in BASE_FACES:
            a = BASE_VERTICES[face[0]]
            b = BASE_VERTICES[face[1]]
            c = BASE_VERTICES[face[2]]
            cross_bc = _cross(b, c)
            cross_ca = _cross(c, a)
            cross_ab = _cross(a, b)
            determinant = _dot(a, cross_bc)
            if abs(determinant) < 1e-12:
                raise SpherePickError("Degenerate base face")
            inverses.append(
                (
                    tuple(value / determinant for value in cross_bc),
                    tuple(value / determinant for value in cross_ca),
                    tuple(value / determinant for value in cross_ab),
                )
            )
        return tuple(inverses)

    def _face_weights(
        self, direction: Sequence[float]
    ) -> tuple[int, tuple[float, float, float]]:
        best_face = 0
        best_weights = (1.0, 0.0, 0.0)
        best_score = -math.inf
        for face_id, rows in enumerate(self._face_inverses):
            weight_a = _dot(rows[0], direction)
            weight_b = _dot(rows[1], direction)
            weight_c = _dot(rows[2], direction)
            total = weight_a + weight_b + weight_c
            if total <= 0.0:
                continue
            score = min(weight_a, weight_b, weight_c)
            if score > best_score:
                best_score = score
                best_face = face_id
                best_weights = (weight_a, weight_b, weight_c)
        if best_score == -math.inf:
            raise SpherePickError("Direction does not fall inside any base face")
        return best_face, best_weights

    def _lattice_weights(self, weights: Sequence[float]) -> tuple[int, int, int]:
        total = weights[0] + weights[1] + weights[2]
        if total <= 0.0:
            raise SpherePickError("Barycentric weights are degenerate")
        scale = self.frequency / total
        scaled = [max(0.0, float(value) * scale) for value in weights]
        integers = [int(math.floor(value)) for value in scaled]
        remainder = self.frequency - sum(integers)
        if remainder > 0:
            order = sorted(
                range(3), key=lambda index: scaled[index] - integers[index], reverse=True
            )
            for index in order[:remainder]:
                integers[index] += 1
        while remainder < 0:
            index = max(range(3), key=lambda item: integers[item] - scaled[item])
            if integers[index] <= 0:
                break
            integers[index] -= 1
            remainder += 1
        # Floating point can still leave the triple one short or one long; fix it
        # on the largest component so ``point_id`` never sees an invalid address.
        difference = self.frequency - sum(integers)
        if difference:
            index = max(range(3), key=lambda item: integers[item])
            integers[index] = max(0, integers[index] + difference)
        return integers[0], integers[1], integers[2]

    def _refine(self, cell_id: int, direction: Sequence[float]) -> int:
        current = int(cell_id)
        current_score = _dot(self.topology.cell_center(current), direction)
        for _step in range(self.REFINE_STEPS):
            best = current
            best_score = current_score
            for neighbor_id in self.topology.cell_neighbor_ids_unordered(current):
                score = _dot(self.topology.cell_center(neighbor_id), direction)
                if score > best_score + 1e-15:
                    best = neighbor_id
                    best_score = score
            if best == current:
                break
            current = best
            current_score = best_score
        return current

    def cell_at_direction(self, direction: Sequence[float]) -> int:
        unit = _normalize(direction)
        face_id, weights = self._face_weights(unit)
        weight_a, weight_b, weight_c = self._lattice_weights(weights)
        try:
            cell_id = self.topology.point_id(face_id, weight_a, weight_b, weight_c)
        except Exception as exc:  # pragma: no cover - defensive
            raise SpherePickError(
                f"Cannot resolve lattice point {face_id}:{weight_a},{weight_b},{weight_c}"
            ) from exc
        return self._refine(cell_id, unit)

    def pick(
        self,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
        radius: float,
        yaw: float,
        pitch: float,
    ) -> int | None:
        direction = screen_to_direction(x, y, center_x, center_y, radius, yaw, pitch)
        if direction is None:
            self.last_cell_id = None
            return None
        cell_id = self.cell_at_direction(direction)
        self.last_cell_id = cell_id
        return cell_id
