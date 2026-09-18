from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .png_pixels import PixelImage
from .production_topology import CellAddress, ProductionTopology
from .topology import BASE_FACES, BASE_VERTICES

SQRT3 = math.sqrt(3.0)
TRIANGLE_HEIGHT = SQRT3 / 2.0
HEX_RADIUS = 1.0 / SQRT3


@dataclass(frozen=True)
class FacePlacement:
    face_id: int
    vertex_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = tuple(point[0] for point in self.vertex_points)
        ys = tuple(point[1] for point in self.vertex_points)
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class FlatCellPolygon:
    face_id: int
    cell_id: int
    points: tuple[float, ...]


class IcosahedralNetLayout:
    """A deterministic five-strip icosahedron net in cell-spacing units.

    Seam vertices and seam cells intentionally appear more than once. Every copy
    references the same CellId, so editing either copy changes the same spherical
    map cell.
    """

    def __init__(self, frequency: int) -> None:
        if frequency < 1:
            raise ValueError("Frequency must be positive")
        self.frequency = int(frequency)
        self.face_placements = self._build_placements()
        self.by_face = {item.face_id: item for item in self.face_placements}
        xs = [value for face in self.face_placements for point in face.vertex_points for value in (point[0],)]
        ys = [value for face in self.face_placements for point in face.vertex_points for value in (point[1],)]
        self.bounds = min(xs), min(ys), max(xs), max(ys)

    def _build_placements(self) -> tuple[FacePlacement, ...]:
        f = float(self.frequency)
        h = TRIANGLE_HEIGHT * f
        upper = (11, 5, 1, 7, 10)
        lower = (4, 9, 8, 6, 2)
        top_faces = (0, 1, 2, 3, 4)
        upper_belt_faces = (6, 5, 9, 8, 7)
        lower_belt_faces = (15, 19, 18, 17, 16)
        bottom_faces = (10, 14, 13, 12, 11)
        result: list[FacePlacement] = []

        def add(face_id: int, coordinates_by_vertex: dict[int, tuple[float, float]]) -> None:
            face = BASE_FACES[face_id]
            result.append(
                FacePlacement(
                    face_id,
                    tuple(coordinates_by_vertex[vertex_id] for vertex_id in face),  # type: ignore[arg-type]
                )
            )

        for index in range(5):
            u0 = upper[index]
            u1 = upper[(index + 1) % 5]
            l0 = lower[index]
            l1 = lower[(index + 1) % 5]
            x = index * f
            top_left = (x, 0.0)
            top_right = (x + f, 0.0)
            lower_left = (x + 0.5 * f, h)
            lower_right = (x + 1.5 * f, h)
            top_apex = (x + 0.5 * f, -h)
            bottom_apex = (x + f, 2.0 * h)

            add(top_faces[index], {0: top_apex, u0: top_left, u1: top_right})
            add(upper_belt_faces[index], {u0: top_left, u1: top_right, l0: lower_left})
            add(lower_belt_faces[index], {u1: top_right, l0: lower_left, l1: lower_right})
            add(bottom_faces[index], {3: bottom_apex, l0: lower_left, l1: lower_right})
        return tuple(sorted(result, key=lambda item: item.face_id))

    def face_point(self, address: CellAddress) -> tuple[float, float]:
        placement = self.by_face[address.face_id]
        total = float(self.frequency)
        weights = (address.weight_a / total, address.weight_b / total, address.weight_c / total)
        return (
            sum(placement.vertex_points[index][0] * weights[index] for index in range(3)),
            sum(placement.vertex_points[index][1] * weights[index] for index in range(3)),
        )

    def barycentric(self, face_id: int, x: float, y: float) -> tuple[float, float, float]:
        a, b, c = self.by_face[int(face_id)].vertex_points
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            raise ValueError("Degenerate face placement")
        wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        wc = 1.0 - wa - wb
        return wa, wb, wc

    def face_at(self, x: float, y: float, *, tolerance: float = 1e-8) -> int | None:
        for placement in self.face_placements:
            left, top, right, bottom = placement.bounds
            if x < left - tolerance or x > right + tolerance or y < top - tolerance or y > bottom + tolerance:
                continue
            weights = self.barycentric(placement.face_id, x, y)
            if min(weights) >= -tolerance:
                return placement.face_id
        return None

    def nearest_cell(self, topology: ProductionTopology, x: float, y: float) -> tuple[int, int] | None:
        face_id = self.face_at(x, y)
        if face_id is None:
            return None
        fractions = self.barycentric(face_id, x, y)
        scaled = [max(0.0, value * self.frequency) for value in fractions]
        integers = [math.floor(value) for value in scaled]
        remainder = self.frequency - sum(integers)
        order = sorted(range(3), key=lambda index: scaled[index] - integers[index], reverse=True)
        for index in order[:remainder]:
            integers[index] += 1
        while sum(integers) > self.frequency:
            index = max(range(3), key=lambda item: integers[item] - scaled[item])
            integers[index] -= 1
        cell_id = topology.point_id(face_id, integers[0], integers[1], integers[2])
        if cell_id < 12:
            return None
        return face_id, cell_id

    def visible_cells(
        self,
        topology: ProductionTopology,
        world_bounds: tuple[float, float, float, float],
        *,
        maximum_cells: int = 8000,
    ) -> tuple[FlatCellPolygon, ...]:
        left, top, right, bottom = world_bounds
        result: list[FlatCellPolygon] = []
        margin = 2
        for placement in self.face_placements:
            face_left, face_top, face_right, face_bottom = placement.bounds
            if right < face_left or left > face_right or bottom < face_top or top > face_bottom:
                continue
            samples = (
                self.barycentric(placement.face_id, left, top),
                self.barycentric(placement.face_id, right, top),
                self.barycentric(placement.face_id, left, bottom),
                self.barycentric(placement.face_id, right, bottom),
            )
            b_values = [value[1] * self.frequency for value in samples]
            c_values = [value[2] * self.frequency for value in samples]
            min_b = max(0, math.floor(min(b_values)) - margin)
            max_b = min(self.frequency, math.ceil(max(b_values)) + margin)
            min_c = max(0, math.floor(min(c_values)) - margin)
            max_c = min(self.frequency, math.ceil(max(c_values)) + margin)
            triangle = placement.vertex_points
            for weight_b in range(min_b, max_b + 1):
                c_upper = min(max_c, self.frequency - weight_b)
                if c_upper < min_c:
                    continue
                for weight_c in range(min_c, c_upper + 1):
                    weight_a = self.frequency - weight_b - weight_c
                    address = CellAddress(placement.face_id, weight_a, weight_b, weight_c)
                    center_x, center_y = self.face_point(address)
                    if center_x < left - 1.0 or center_x > right + 1.0 or center_y < top - 1.0 or center_y > bottom + 1.0:
                        continue
                    cell_id = topology.point_id(placement.face_id, weight_a, weight_b, weight_c)
                    if cell_id < 12:
                        continue
                    polygon = _clip_polygon(_regular_hex(center_x, center_y), triangle)
                    if len(polygon) < 3:
                        continue
                    flat = tuple(value for point in polygon for value in point)
                    result.append(FlatCellPolygon(placement.face_id, cell_id, flat))
                    if len(result) > maximum_cells:
                        return ()
        return tuple(result)


def _regular_hex(center_x: float, center_y: float) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            center_x + HEX_RADIUS * math.cos(math.radians(30.0 + index * 60.0)),
            center_y + HEX_RADIUS * math.sin(math.radians(30.0 + index * 60.0)),
        )
        for index in range(6)
    )


def _signed_area(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _clip_polygon(
    polygon: Iterable[tuple[float, float]],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    output = list(polygon)
    orientation = 1.0 if _signed_area(triangle[0], triangle[1], triangle[2]) >= 0.0 else -1.0
    for edge_index in range(3):
        edge_a = triangle[edge_index]
        edge_b = triangle[(edge_index + 1) % 3]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = orientation * _signed_area(edge_a, edge_b, previous) >= -1e-9
        for current in input_points:
            current_inside = orientation * _signed_area(edge_a, edge_b, current) >= -1e-9
            if current_inside != previous_inside:
                output.append(_line_intersection(previous, current, edge_a, edge_b))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    return tuple(output)


def _line_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
) -> tuple[float, float]:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    ex = edge_b[0] - edge_a[0]
    ey = edge_b[1] - edge_a[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) < 1e-12:
        return second
    t = ((edge_a[0] - first[0]) * ey - (edge_a[1] - first[1]) * ex) / denominator
    return first[0] + t * dx, first[1] + t * dy


def render_net_surface_ppm(
    texture: PixelImage,
    net: IcosahedralNetLayout,
    center_world_x: float,
    center_world_y: float,
    view_scale: float,
    width: int,
    height: int,
    *,
    block_size: int = 2,
    background: tuple[int, int, int] = (20, 25, 31),
) -> bytes:
    if texture.channels not in (3, 4):
        raise ValueError("Flat-map texture must be RGB or RGBA")
    width = max(1, int(width))
    height = max(1, int(height))
    view_scale = max(1e-6, float(view_scale))
    block_size = max(1, int(block_size))
    output = bytearray(bytes(background) * (width * height))
    source = texture.pixels
    channels = texture.channels
    for placement in net.face_placements:
        screen = tuple(
            (
                (point[0] - center_world_x) * view_scale + width / 2.0,
                (point[1] - center_world_y) * view_scale + height / 2.0,
            )
            for point in placement.vertex_points
        )
        min_x = max(0, int(math.floor(min(point[0] for point in screen))))
        max_x = min(width - 1, int(math.ceil(max(point[0] for point in screen))))
        min_y = max(0, int(math.floor(min(point[1] for point in screen))))
        max_y = min(height - 1, int(math.ceil(max(point[1] for point in screen))))
        if min_x > max_x or min_y > max_y:
            continue
        face = BASE_FACES[placement.face_id]
        vertices = tuple(BASE_VERTICES[vertex_id] for vertex_id in face)
        for y0 in range(min_y, max_y + 1, block_size):
            sample_y = min(height - 0.5, y0 + block_size * 0.5)
            world_y = center_world_y + (sample_y - height / 2.0) / view_scale
            for x0 in range(min_x, max_x + 1, block_size):
                sample_x = min(width - 0.5, x0 + block_size * 0.5)
                world_x = center_world_x + (sample_x - width / 2.0) / view_scale
                wa, wb, wc = net.barycentric(placement.face_id, world_x, world_y)
                if min(wa, wb, wc) < -1e-8:
                    continue
                px = vertices[0][0] * wa + vertices[1][0] * wb + vertices[2][0] * wc
                py = vertices[0][1] * wa + vertices[1][1] * wb + vertices[2][1] * wc
                pz = vertices[0][2] * wa + vertices[1][2] * wb + vertices[2][2] * wc
                length = math.sqrt(px * px + py * py + pz * pz)
                px, py, pz = px / length, py / length, pz / length
                longitude = math.atan2(pz, px)
                latitude = math.asin(max(-1.0, min(1.0, py)))
                source_x = int((longitude + math.pi) / (2.0 * math.pi) * texture.width) % texture.width
                source_y = min(texture.height - 1, max(0, int((math.pi / 2.0 - latitude) / math.pi * texture.height)))
                offset = (source_y * texture.width + source_x) * channels
                red, green, blue = source[offset], source[offset + 1], source[offset + 2]
                if channels == 4:
                    alpha = source[offset + 3] / 255.0
                    red = round(red * alpha + background[0] * (1.0 - alpha))
                    green = round(green * alpha + background[1] * (1.0 - alpha))
                    blue = round(blue * alpha + background[2] * (1.0 - alpha))
                payload = bytes((red, green, blue)) * min(block_size, width - x0)
                for target_y in range(y0, min(height, y0 + block_size)):
                    target_offset = (target_y * width + x0) * 3
                    output[target_offset:target_offset + len(payload)] = payload
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(output)
