from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .brush_lod import LOD_EFFECTIVE_SIZES, LOD_PADDING
from .gpu_batch import rotate_uv
from .png_pixels import PixelImage, atomic_write_png


@dataclass(frozen=True)
class SeamValidationReport:
    valid: bool
    sampled_cells: int
    checked_neighbor_pairs: int
    max_shared_corner_error: float
    max_rotation_round_trip_error: float
    uv_inside_padded_unit: bool
    issues: tuple[str, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def validate_topology_seams(
    topology,
    cell_ids: Iterable[int] | None = None,
    *,
    sample_limit: int = 4096,
    tolerance: float = 1e-9,
) -> SeamValidationReport:
    if cell_ids is None:
        count = topology.cell_count
        step = max(1, count // max(1, sample_limit))
        requested = list(range(0, count, step))[:sample_limit]
        requested.extend(topology.pentagon_ids)
        cell_ids = tuple(dict.fromkeys(requested))
    sampled = tuple(int(cell_id) for cell_id in cell_ids)
    issues: list[str] = []
    max_error = 0.0
    checked = 0
    visited_pairs: set[tuple[int, int]] = set()
    for cell_id in sampled:
        corners = topology.cell_corners(cell_id) if hasattr(topology, "cell_corners") else tuple(
            topology.triangle_centers[index] for index in topology.incident_triangles[cell_id]
        )
        neighbors = topology.cell_neighbors(cell_id) if hasattr(topology, "cell_neighbors") else topology.neighbors[cell_id]
        for neighbor_id in neighbors:
            pair = tuple(sorted((cell_id, int(neighbor_id))))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            other = topology.cell_corners(neighbor_id) if hasattr(topology, "cell_corners") else tuple(
                topology.triangle_centers[index] for index in topology.incident_triangles[neighbor_id]
            )
            matches = sorted(
                _distance(first, second)
                for first in corners
                for second in other
            )[:2]
            if len(matches) != 2:
                issues.append(f"CellId {cell_id} / {neighbor_id} has no shared edge")
                continue
            pair_error = max(matches)
            max_error = max(max_error, pair_error)
            if pair_error > tolerance:
                issues.append(
                    f"CellId {cell_id} / {neighbor_id} shared corner error {pair_error:.3e}"
                )
            checked += 1

    uv_points = [(0.5, 0.5)] + [
        (0.5 + math.cos(math.radians(index * 60)) * 0.5,
         0.5 - math.sin(math.radians(index * 60)) * 0.5)
        for index in range(6)
    ]
    max_round_trip = 0.0
    uv_inside = True
    for level, effective in enumerate(LOD_EFFECTIVE_SIZES):
        padded = effective + LOD_PADDING * 2
        ratio = effective / padded
        lower = LOD_PADDING / padded - 1e-12
        upper = 1.0 - LOD_PADDING / padded + 1e-12
        for uv in uv_points:
            current = uv
            for rotation in range(6):
                rotated = rotate_uv(uv, rotation)
                padded_uv = (
                    0.5 + (rotated[0] - 0.5) * ratio,
                    0.5 + (rotated[1] - 0.5) * ratio,
                )
                if not (lower <= padded_uv[0] <= upper and lower <= padded_uv[1] <= upper):
                    uv_inside = False
                    issues.append(f"LOD{level} rotation {rotation} UV escaped padding")
            for _ in range(6):
                current = rotate_uv(current, 1)
            max_round_trip = max(
                max_round_trip,
                math.hypot(current[0] - uv[0], current[1] - uv[1]),
            )
    if max_round_trip > 1e-12:
        issues.append(f"Six-rotation UV round trip error {max_round_trip:.3e}")
    return SeamValidationReport(
        valid=not issues,
        sampled_cells=len(sampled),
        checked_neighbor_pairs=checked,
        max_shared_corner_error=max_error,
        max_rotation_round_trip_error=max_round_trip,
        uv_inside_padded_unit=uv_inside,
        issues=tuple(issues),
    )


def create_seam_test_brushes(directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    edge_colors = (
        (255, 64, 64),
        (255, 192, 64),
        (96, 224, 96),
        (64, 192, 255),
        (128, 96, 255),
        (255, 96, 224),
    )
    size = 512
    for rotation, base in enumerate(edge_colors):
        pixels = bytearray(size * size * 3)
        for y in range(size):
            for x in range(size):
                border = min(x, y, size - 1 - x, size - 1 - y)
                if border < 8:
                    color = edge_colors[(rotation + (x // 64 + y // 64)) % 6]
                else:
                    color = base
                offset = (y * size + x) * 3
                pixels[offset : offset + 3] = bytes(color)
        path = directory / f"seam_rotation_{rotation}.png"
        atomic_write_png(path, PixelImage(size, size, 3, bytes(pixels)))
        paths.append(path)
    return tuple(paths)


def _distance(first, second) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))
