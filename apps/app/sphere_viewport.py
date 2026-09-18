from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .chunk_layout import ChunkLayout
from .chunk_visibility import ChunkVisibilityIndex
from .topology import DualTopology


@dataclass(frozen=True)
class ProjectedCell:
    cell_id: int
    depth: float
    center_x: float
    center_y: float
    points: tuple[float, ...]
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    pixel_radius: float

    def contains(self, x: float, y: float) -> bool:
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
        vertices = tuple(zip(self.points[0::2], self.points[1::2]))
        inside = False
        previous_x, previous_y = vertices[-1]
        for current_x, current_y in vertices:
            intersects = (current_y > y) != (previous_y > y)
            if intersects:
                denominator = previous_y - current_y
                if denominator == 0.0:
                    denominator = 1e-12
                boundary_x = (previous_x - current_x) * (y - current_y) / denominator + current_x
                if x < boundary_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside


@dataclass(frozen=True)
class ProjectedChunk:
    chunk_id: int
    depth: float
    center_x: float
    center_y: float
    points: tuple[float, ...]
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class ViewportProjection:
    cells: tuple[ProjectedCell, ...]
    visible_cell_ids: tuple[int, ...]
    visible_chunk_ids: tuple[int, ...]
    sphere_center_x: float
    sphere_center_y: float
    sphere_radius: float
    chunks: tuple[ProjectedChunk, ...] = ()
    candidate_cell_count: int = 0
    visited_visibility_nodes: int = 0
    tested_visibility_chunks: int = 0

    def hit_test(self, x: float, y: float) -> int | None:
        for cell in reversed(self.cells):
            if cell.contains(x, y):
                return cell.cell_id
        return None


class SphereViewport:
    def __init__(self, yaw: float = -0.35, pitch: float = 0.25, zoom: float = 1.0) -> None:
        self.yaw = yaw
        self.pitch = pitch
        self.zoom = zoom

    def rotate_by(self, delta_x: float, delta_y: float) -> None:
        self.yaw += delta_x * 0.008
        self.pitch = max(-1.45, min(1.45, self.pitch + delta_y * 0.008))

    def zoom_by(self, steps: float) -> None:
        self.zoom = max(0.8, min(12.0, self.zoom * (1.16**steps)))

    @staticmethod
    def rotate_point(
        point: tuple[float, float, float], yaw: float, pitch: float
    ) -> tuple[float, float, float]:
        x, y, z = point
        cosine_yaw = math.cos(yaw)
        sine_yaw = math.sin(yaw)
        x, z = x * cosine_yaw + z * sine_yaw, -x * sine_yaw + z * cosine_yaw
        cosine_pitch = math.cos(pitch)
        sine_pitch = math.sin(pitch)
        y, z = y * cosine_pitch - z * sine_pitch, y * sine_pitch + z * cosine_pitch
        return x, y, z

    def project(
        self,
        topology: DualTopology,
        width: int,
        height: int,
        layout: ChunkLayout | None = None,
        margin: float = 12.0,
        cell_ids: Iterable[int] | None = None,
        candidate_cell_count: int | None = None,
        visited_visibility_nodes: int = 0,
        tested_visibility_chunks: int = 0,
    ) -> ViewportProjection:
        width = max(1, width)
        height = max(1, height)
        base_radius = max(20.0, min(width, height) * 0.43)
        radius = base_radius * self.zoom
        center_x = width / 2.0
        center_y = height / 2.0

        if cell_ids is None:
            requested_ids: Iterable[int] = range(topology.cell_count)
            requested_count = topology.cell_count
        else:
            unique_ids = tuple(dict.fromkeys(int(cell_id) for cell_id in cell_ids))
            invalid = [cell_id for cell_id in unique_ids if cell_id < 0 or cell_id >= topology.cell_count]
            if invalid:
                raise IndexError(f"CellId is outside topology: {invalid[0]}")
            requested_ids = unique_ids
            requested_count = len(unique_ids)

        rotated_corner_cache: dict[int, tuple[float, float, float]] = {}
        projected: list[ProjectedCell] = []
        for cell_id in requested_ids:
            cell_x, cell_y, cell_z = self.rotate_point(
                topology.cell_centers[cell_id], self.yaw, self.pitch
            )
            if cell_z <= 0.0:
                continue
            point_values: list[float] = []
            corner_depths: list[float] = []
            xs: list[float] = []
            ys: list[float] = []
            for corner_id in topology.incident_triangles[cell_id]:
                rotated = rotated_corner_cache.get(corner_id)
                if rotated is None:
                    rotated = self.rotate_point(
                        topology.triangle_centers[corner_id], self.yaw, self.pitch
                    )
                    rotated_corner_cache[corner_id] = rotated
                corner_x, corner_y, corner_z = rotated
                screen_x = center_x + corner_x * radius
                screen_y = center_y - corner_y * radius
                point_values.extend((screen_x, screen_y))
                xs.append(screen_x)
                ys.append(screen_y)
                corner_depths.append(corner_z)
            if not xs:
                continue
            if sum(depth >= -0.04 for depth in corner_depths) < len(corner_depths) - 1:
                continue
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            if max_x < -margin or min_x > width + margin or max_y < -margin or min_y > height + margin:
                continue
            screen_center_x = center_x + cell_x * radius
            screen_center_y = center_y - cell_y * radius
            pixel_radius = max(max_x - min_x, max_y - min_y) / 2.0
            projected.append(
                ProjectedCell(
                    cell_id=cell_id,
                    depth=cell_z,
                    center_x=screen_center_x,
                    center_y=screen_center_y,
                    points=tuple(point_values),
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    pixel_radius=pixel_radius,
                )
            )

        projected.sort(key=lambda cell: cell.depth)
        visible_cell_ids = tuple(cell.cell_id for cell in projected)
        if layout is None:
            visible_chunk_ids: tuple[int, ...] = ()
        else:
            visible_chunk_ids = tuple(
                sorted({layout.cell_to_chunk[cell_id] for cell_id in visible_cell_ids})
            )
        return ViewportProjection(
            cells=tuple(projected),
            visible_cell_ids=visible_cell_ids,
            visible_chunk_ids=visible_chunk_ids,
            sphere_center_x=center_x,
            sphere_center_y=center_y,
            sphere_radius=radius,
            candidate_cell_count=(requested_count if candidate_cell_count is None else candidate_cell_count),
            visited_visibility_nodes=visited_visibility_nodes,
            tested_visibility_chunks=tested_visibility_chunks,
        )

    def project_chunks(
        self,
        topology: DualTopology,
        visibility: ChunkVisibilityIndex,
        chunk_ids: Iterable[int],
        width: int,
        height: int,
        margin: float = 12.0,
        candidate_cell_count: int = 0,
        visited_visibility_nodes: int = 0,
        tested_visibility_chunks: int = 0,
    ) -> ViewportProjection:
        width = max(1, width)
        height = max(1, height)
        base_radius = max(20.0, min(width, height) * 0.43)
        radius = base_radius * self.zoom
        center_x = width / 2.0
        center_y = height / 2.0
        corner_cache: dict[int, tuple[float, float, float]] = {}
        projected_chunks: list[ProjectedChunk] = []

        for chunk_id in chunk_ids:
            if chunk_id < 0 or chunk_id >= visibility.chunk_count:
                raise IndexError(f"ChunkId is outside visibility index: {chunk_id}")
            bound = visibility.chunks[chunk_id]
            rotated_center = self.rotate_point(bound.center, self.yaw, self.pitch)
            points: list[tuple[float, float]] = []
            depths: list[float] = []
            for triangle_id in bound.boundary_triangle_ids:
                rotated = corner_cache.get(triangle_id)
                if rotated is None:
                    rotated = self.rotate_point(
                        topology.triangle_centers[triangle_id], self.yaw, self.pitch
                    )
                    corner_cache[triangle_id] = rotated
                x, y, z = rotated
                if z < -0.04:
                    continue
                points.append((center_x + x * radius, center_y - y * radius))
                depths.append(z)
            if len(points) < 3:
                continue
            hull = _convex_hull(points)
            if len(hull) < 3:
                continue
            xs = [point[0] for point in hull]
            ys = [point[1] for point in hull]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            if max_x < -margin or min_x > width + margin or max_y < -margin or min_y > height + margin:
                continue
            point_values = tuple(value for point in hull for value in point)
            projected_chunks.append(
                ProjectedChunk(
                    chunk_id=chunk_id,
                    depth=sum(depths) / len(depths),
                    center_x=center_x + rotated_center[0] * radius,
                    center_y=center_y - rotated_center[1] * radius,
                    points=point_values,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                )
            )

        projected_chunks.sort(key=lambda chunk: chunk.depth)
        visible_chunk_ids = tuple(chunk.chunk_id for chunk in projected_chunks)
        return ViewportProjection(
            cells=(),
            visible_cell_ids=(),
            visible_chunk_ids=visible_chunk_ids,
            sphere_center_x=center_x,
            sphere_center_y=center_y,
            sphere_radius=radius,
            chunks=tuple(projected_chunks),
            candidate_cell_count=candidate_cell_count,
            visited_visibility_nodes=visited_visibility_nodes,
            tested_visibility_chunks=tested_visibility_chunks,
        )


def chunk_ids_for_cells(layout: ChunkLayout, cell_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({layout.cell_to_chunk[cell_id] for cell_id in cell_ids}))


def _convex_hull(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return tuple(unique)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])
