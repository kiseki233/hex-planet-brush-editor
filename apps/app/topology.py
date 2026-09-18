from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

GENERATOR_VERSION = 1
ORIENTATION = "default_v1"
TOPOLOGY_TYPE = "icosahedral_dual"
CELL_MAGIC = b"HPCELL1\0"
TRIANGLE_MAGIC = b"HPTRI01\0"
BINARY_VERSION = 1
MISSING_INDEX = 0xFFFFFFFF
CELL_HEADER = struct.Struct("<8sIIII")
CELL_RECORD = struct.Struct("<3fBBH6I6I")
TRIANGLE_HEADER = struct.Struct("<8sIIII")
TRIANGLE_RECORD = struct.Struct("<3f")


class TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class TopologyValidation:
    valid: bool
    cell_count: int
    triangle_count: int
    edge_count: int
    pentagon_count: int
    hexagon_count: int
    euler_characteristic: int
    reciprocal_neighbor_links: bool
    stable_hash: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class DualTopology:
    frequency: int
    cell_centers: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    triangle_centers: tuple[tuple[float, float, float], ...]
    neighbors: tuple[tuple[int, ...], ...]
    incident_triangles: tuple[tuple[int, ...], ...]
    pentagon_ids: tuple[int, ...]
    stable_hash: str

    @property
    def cell_count(self) -> int:
        return len(self.cell_centers)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @property
    def edge_count(self) -> int:
        return sum(len(items) for items in self.neighbors) // 2

    @property
    def hexagon_count(self) -> int:
        return self.cell_count - len(self.pentagon_ids)

    def validate(self) -> TopologyValidation:
        issues: list[str] = []
        expected_cells = 10 * self.frequency * self.frequency + 2
        expected_triangles = 20 * self.frequency * self.frequency
        expected_edges = 30 * self.frequency * self.frequency

        if self.cell_count != expected_cells:
            issues.append(f"cell count mismatch: expected {expected_cells}, received {self.cell_count}")
        if self.triangle_count != expected_triangles:
            issues.append(
                f"triangle count mismatch: expected {expected_triangles}, received {self.triangle_count}"
            )
        if self.edge_count != expected_edges:
            issues.append(f"edge count mismatch: expected {expected_edges}, received {self.edge_count}")
        if len(self.pentagon_ids) != 12:
            issues.append(f"pentagon count mismatch: expected 12, received {len(self.pentagon_ids)}")

        pentagon_set = set(self.pentagon_ids)
        reciprocal = True
        for cell_id, cell_neighbors in enumerate(self.neighbors):
            expected_degree = 5 if cell_id in pentagon_set else 6
            if len(cell_neighbors) != expected_degree:
                issues.append(
                    f"cell {cell_id} degree mismatch: expected {expected_degree}, received {len(cell_neighbors)}"
                )
            if len(self.incident_triangles[cell_id]) != expected_degree:
                issues.append(
                    f"cell {cell_id} incident triangle mismatch: expected {expected_degree}, "
                    f"received {len(self.incident_triangles[cell_id])}"
                )
            for neighbor_id in cell_neighbors:
                if cell_id not in self.neighbors[neighbor_id]:
                    reciprocal = False
                    issues.append(f"neighbor link is not reciprocal: {cell_id} -> {neighbor_id}")
                    break

        euler = self.cell_count - self.edge_count + self.triangle_count
        if euler != 2:
            issues.append(f"Euler characteristic mismatch: expected 2, received {euler}")

        return TopologyValidation(
            valid=not issues,
            cell_count=self.cell_count,
            triangle_count=self.triangle_count,
            edge_count=self.edge_count,
            pentagon_count=len(self.pentagon_ids),
            hexagon_count=self.hexagon_count,
            euler_characteristic=euler,
            reciprocal_neighbor_links=reciprocal,
            stable_hash=self.stable_hash,
            issues=tuple(issues),
        )


@dataclass(frozen=True)
class TopologyCacheInfo:
    directory: Path
    manifest_path: Path
    cells_path: Path
    triangles_path: Path
    cell_crc32: str
    triangle_crc32: str
    cell_sha256: str
    triangle_sha256: str


_BASE_VERTICES_RAW: tuple[tuple[float, float, float], ...] = (
    (-1.0, (1.0 + math.sqrt(5.0)) / 2.0, 0.0),
    (1.0, (1.0 + math.sqrt(5.0)) / 2.0, 0.0),
    (-1.0, -(1.0 + math.sqrt(5.0)) / 2.0, 0.0),
    (1.0, -(1.0 + math.sqrt(5.0)) / 2.0, 0.0),
    (0.0, -1.0, (1.0 + math.sqrt(5.0)) / 2.0),
    (0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0),
    (0.0, -1.0, -(1.0 + math.sqrt(5.0)) / 2.0),
    (0.0, 1.0, -(1.0 + math.sqrt(5.0)) / 2.0),
    ((1.0 + math.sqrt(5.0)) / 2.0, 0.0, -1.0),
    ((1.0 + math.sqrt(5.0)) / 2.0, 0.0, 1.0),
    (-(1.0 + math.sqrt(5.0)) / 2.0, 0.0, -1.0),
    (-(1.0 + math.sqrt(5.0)) / 2.0, 0.0, 1.0),
)

_BASE_FACES_RAW: tuple[tuple[int, int, int], ...] = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _subtract(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(vector: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return vector[0] * value, vector[1] * value, vector[2] * value


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length == 0.0:
        raise TopologyError("Cannot normalize a zero-length vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


def _weighted_position(
    vertex_a: tuple[float, float, float],
    vertex_b: tuple[float, float, float],
    vertex_c: tuple[float, float, float],
    weight_a: int,
    weight_b: int,
    weight_c: int,
) -> tuple[float, float, float]:
    return _normalize(
        (
            vertex_a[0] * weight_a + vertex_b[0] * weight_b + vertex_c[0] * weight_c,
            vertex_a[1] * weight_a + vertex_b[1] * weight_b + vertex_c[1] * weight_c,
            vertex_a[2] * weight_a + vertex_b[2] * weight_b + vertex_c[2] * weight_c,
        )
    )


def _orient_base_faces(
    vertices: tuple[tuple[float, float, float], ...],
    faces: Iterable[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    oriented: list[tuple[int, int, int]] = []
    for first, second, third in faces:
        a = vertices[first]
        b = vertices[second]
        c = vertices[third]
        normal = _cross(_subtract(b, a), _subtract(c, a))
        centroid = _normalize(_add(_add(a, b), c))
        if _dot(normal, centroid) < 0.0:
            oriented.append((first, third, second))
        else:
            oriented.append((first, second, third))
    return tuple(oriented)


BASE_VERTICES = tuple(_normalize(vertex) for vertex in _BASE_VERTICES_RAW)
BASE_FACES = _orient_base_faces(BASE_VERTICES, _BASE_FACES_RAW)
BASE_EDGES = tuple(
    sorted(
        {
            tuple(sorted((face[index], face[(index + 1) % 3])))
            for face in BASE_FACES
            for index in range(3)
        }
    )
)
BASE_EDGE_INDEX = {edge: index for index, edge in enumerate(BASE_EDGES)}


class _PointIndexer:
    def __init__(self, frequency: int) -> None:
        self.frequency = frequency
        self.edge_point_count = max(0, frequency - 1)
        self.interior_per_face = max(0, (frequency - 1) * (frequency - 2) // 2)
        self.edge_start = 12
        self.face_start = self.edge_start + len(BASE_EDGES) * self.edge_point_count

    @property
    def total_count(self) -> int:
        return 10 * self.frequency * self.frequency + 2

    def point_id(self, face_index: int, weight_a: int, weight_b: int, weight_c: int) -> int:
        frequency = self.frequency
        if weight_a + weight_b + weight_c != frequency:
            raise TopologyError("Barycentric weights do not match frequency")
        if min(weight_a, weight_b, weight_c) < 0:
            raise TopologyError("Barycentric weights must not be negative")

        a, b, c = BASE_FACES[face_index]
        if weight_a == frequency:
            return a
        if weight_b == frequency:
            return b
        if weight_c == frequency:
            return c
        if weight_c == 0:
            return self._edge_point_id(a, b, weight_a, weight_b)
        if weight_b == 0:
            return self._edge_point_id(a, c, weight_a, weight_c)
        if weight_a == 0:
            return self._edge_point_id(b, c, weight_b, weight_c)

        local_i = weight_b
        local_j = weight_c
        rank = (local_i - 1) * (frequency - 1) - ((local_i - 1) * local_i) // 2 + (local_j - 1)
        return self.face_start + face_index * self.interior_per_face + rank

    def _edge_point_id(self, first: int, second: int, first_weight: int, second_weight: int) -> int:
        low, high = sorted((first, second))
        high_weight = second_weight if second == high else first_weight
        if high_weight <= 0 or high_weight >= self.frequency:
            raise TopologyError("Edge point weight is outside the interior edge range")
        edge_index = BASE_EDGE_INDEX[(low, high)]
        return self.edge_start + edge_index * self.edge_point_count + high_weight - 1


def _build_cell_centers(frequency: int, indexer: _PointIndexer) -> list[tuple[float, float, float]]:
    centers: list[tuple[float, float, float]] = list(BASE_VERTICES)

    for low, high in BASE_EDGES:
        low_vertex = BASE_VERTICES[low]
        high_vertex = BASE_VERTICES[high]
        for high_weight in range(1, frequency):
            low_weight = frequency - high_weight
            centers.append(
                _normalize(
                    (
                        low_vertex[0] * low_weight + high_vertex[0] * high_weight,
                        low_vertex[1] * low_weight + high_vertex[1] * high_weight,
                        low_vertex[2] * low_weight + high_vertex[2] * high_weight,
                    )
                )
            )

    if frequency >= 3:
        for face in BASE_FACES:
            a, b, c = (BASE_VERTICES[index] for index in face)
            for weight_b in range(1, frequency - 1):
                for weight_c in range(1, frequency - weight_b):
                    weight_a = frequency - weight_b - weight_c
                    centers.append(_weighted_position(a, b, c, weight_a, weight_b, weight_c))

    if len(centers) != indexer.total_count:
        raise TopologyError(
            f"Generated cell center count mismatch: expected {indexer.total_count}, received {len(centers)}"
        )
    return centers


def _triangle_is_outward(
    triangle: tuple[int, int, int], centers: list[tuple[float, float, float]]
) -> bool:
    first, second, third = (centers[index] for index in triangle)
    normal = _cross(_subtract(second, first), _subtract(third, first))
    centroid = _normalize(_add(_add(first, second), third))
    return _dot(normal, centroid) > 0.0


def _build_triangles(
    frequency: int,
    indexer: _PointIndexer,
    centers: list[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face_index in range(len(BASE_FACES)):
        for weight_b in range(0, frequency):
            for weight_c in range(0, frequency - weight_b):
                weight_a = frequency - weight_b - weight_c
                first = indexer.point_id(face_index, weight_a, weight_b, weight_c)
                second = indexer.point_id(face_index, weight_a - 1, weight_b + 1, weight_c)
                third = indexer.point_id(face_index, weight_a - 1, weight_b, weight_c + 1)
                triangle = (first, second, third)
                if not _triangle_is_outward(triangle, centers):
                    triangle = (first, third, second)
                triangles.append(triangle)

                if weight_a >= 2:
                    fourth = indexer.point_id(face_index, weight_a - 2, weight_b + 1, weight_c + 1)
                    triangle = (second, fourth, third)
                    if not _triangle_is_outward(triangle, centers):
                        triangle = (second, third, fourth)
                    triangles.append(triangle)

    expected = 20 * frequency * frequency
    if len(triangles) != expected:
        raise TopologyError(f"Generated triangle count mismatch: expected {expected}, received {len(triangles)}")
    return triangles


def _local_basis(
    center: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    reference = (0.0, 0.0, 1.0) if abs(center[2]) < 0.9 else (1.0, 0.0, 0.0)
    tangent_x = _normalize(_cross(reference, center))
    tangent_y = _normalize(_cross(center, tangent_x))
    return tangent_x, tangent_y


def _angle_around(
    center: tuple[float, float, float],
    point: tuple[float, float, float],
    tangent_x: tuple[float, float, float],
    tangent_y: tuple[float, float, float],
) -> float:
    projected = _subtract(point, _scale(center, _dot(point, center)))
    return math.atan2(_dot(projected, tangent_y), _dot(projected, tangent_x))


def _build_adjacency(
    centers: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    tuple[int, ...],
]:
    neighbor_sets: list[set[int]] = [set() for _ in centers]
    incident_lists: list[list[int]] = [[] for _ in centers]
    triangle_centers: list[tuple[float, float, float]] = []

    for triangle_id, (first, second, third) in enumerate(triangles):
        for cell_id in (first, second, third):
            incident_lists[cell_id].append(triangle_id)
        neighbor_sets[first].update((second, third))
        neighbor_sets[second].update((first, third))
        neighbor_sets[third].update((first, second))
        triangle_centers.append(_normalize(_add(_add(centers[first], centers[second]), centers[third])))

    ordered_neighbors: list[tuple[int, ...]] = []
    ordered_incident: list[tuple[int, ...]] = []
    pentagons: list[int] = []

    for cell_id, center in enumerate(centers):
        tangent_x, tangent_y = _local_basis(center)
        neighbors = tuple(
            sorted(
                neighbor_sets[cell_id],
                key=lambda neighbor_id: _angle_around(
                    center, centers[neighbor_id], tangent_x, tangent_y
                ),
            )
        )
        incident = tuple(
            sorted(
                incident_lists[cell_id],
                key=lambda triangle_id: _angle_around(
                    center, triangle_centers[triangle_id], tangent_x, tangent_y
                ),
            )
        )
        ordered_neighbors.append(neighbors)
        ordered_incident.append(incident)
        if len(neighbors) == 5:
            pentagons.append(cell_id)

    return triangle_centers, ordered_neighbors, ordered_incident, tuple(pentagons)


def _stable_topology_hash(
    frequency: int,
    centers: Iterable[tuple[float, float, float]],
    triangles: Iterable[tuple[int, int, int]],
    neighbors: Iterable[tuple[int, ...]],
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<II", GENERATOR_VERSION, frequency))
    for center in centers:
        digest.update(struct.pack("<3d", *center))
    for triangle in triangles:
        digest.update(struct.pack("<3I", *triangle))
    for cell_neighbors in neighbors:
        digest.update(struct.pack("<B", len(cell_neighbors)))
        for neighbor_id in cell_neighbors:
            digest.update(struct.pack("<I", neighbor_id))
    return digest.hexdigest()


def generate_dual_topology(frequency: int, max_cells: int | None = None) -> DualTopology:
    if frequency < 1:
        raise TopologyError("Frequency must be at least 1")
    expected_cells = 10 * frequency * frequency + 2
    if max_cells is not None and expected_cells > max_cells:
        raise TopologyError(
            f"Requested topology has {expected_cells} cells, which exceeds the configured limit of {max_cells}"
        )

    indexer = _PointIndexer(frequency)
    centers = _build_cell_centers(frequency, indexer)
    triangles = _build_triangles(frequency, indexer, centers)
    triangle_centers, neighbors, incident, pentagons = _build_adjacency(centers, triangles)
    stable_hash = _stable_topology_hash(frequency, centers, triangles, neighbors)

    topology = DualTopology(
        frequency=frequency,
        cell_centers=tuple(centers),
        triangles=tuple(triangles),
        triangle_centers=tuple(triangle_centers),
        neighbors=tuple(neighbors),
        incident_triangles=tuple(incident),
        pentagon_ids=pentagons,
        stable_hash=stable_hash,
    )
    validation = topology.validate()
    if not validation.valid:
        raise TopologyError("Topology validation failed: " + "; ".join(validation.issues[:8]))
    return topology


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_topology_cache(topology: DualTopology, map_root: str | Path) -> TopologyCacheInfo:
    validation = topology.validate()
    if not validation.valid:
        raise TopologyError("Cannot write an invalid topology")

    topology_root = Path(map_root) / ".topology"
    directory = topology_root / f"ico_dual_f{topology.frequency}_v{GENERATOR_VERSION}"
    directory.mkdir(parents=True, exist_ok=True)
    cells_path = directory / "cells.bin"
    triangles_path = directory / "triangles.bin"
    manifest_path = directory / "topology.json"

    cell_payload = bytearray(
        CELL_HEADER.pack(
            CELL_MAGIC,
            BINARY_VERSION,
            topology.frequency,
            topology.cell_count,
            topology.triangle_count,
        )
    )
    pentagon_set = set(topology.pentagon_ids)
    for cell_id in range(topology.cell_count):
        neighbors = list(topology.neighbors[cell_id])
        incident = list(topology.incident_triangles[cell_id])
        degree = len(neighbors)
        neighbors.extend([MISSING_INDEX] * (6 - degree))
        incident.extend([MISSING_INDEX] * (6 - len(incident)))
        flags = 1 if cell_id in pentagon_set else 0
        cell_payload.extend(
            CELL_RECORD.pack(
                *topology.cell_centers[cell_id],
                degree,
                flags,
                0,
                *neighbors,
                *incident,
            )
        )

    triangle_payload = bytearray(
        TRIANGLE_HEADER.pack(
            TRIANGLE_MAGIC,
            BINARY_VERSION,
            topology.frequency,
            topology.cell_count,
            topology.triangle_count,
        )
    )
    for center in topology.triangle_centers:
        triangle_payload.extend(TRIANGLE_RECORD.pack(*center))

    _atomic_write_bytes(cells_path, bytes(cell_payload))
    _atomic_write_bytes(triangles_path, bytes(triangle_payload))

    cell_crc = f"{zlib.crc32(cell_payload) & 0xFFFFFFFF:08x}"
    triangle_crc = f"{zlib.crc32(triangle_payload) & 0xFFFFFFFF:08x}"
    cell_sha = hashlib.sha256(cell_payload).hexdigest()
    triangle_sha = hashlib.sha256(triangle_payload).hexdigest()
    payload = {
        "format": "HEX_PLANET_TOPOLOGY_CACHE",
        "formatVersion": 1,
        "topology": {
            "type": TOPOLOGY_TYPE,
            "frequency": topology.frequency,
            "generatorVersion": GENERATOR_VERSION,
            "orientation": ORIENTATION,
            "unitSphere": True,
        },
        "counts": {
            "cells": topology.cell_count,
            "triangles": topology.triangle_count,
            "edges": topology.edge_count,
            "pentagons": len(topology.pentagon_ids),
            "hexagons": topology.hexagon_count,
        },
        "stableHash": topology.stable_hash,
        "files": {
            "cells": {
                "path": "cells.bin",
                "recordBytes": CELL_RECORD.size,
                "crc32": cell_crc,
                "sha256": cell_sha,
            },
            "triangles": {
                "path": "triangles.bin",
                "recordBytes": TRIANGLE_RECORD.size,
                "crc32": triangle_crc,
                "sha256": triangle_sha,
            },
        },
        "cellRecord": {
            "cellId": "implicit_record_index",
            "center": "float32_xyz_unit_sphere",
            "degree": "uint8",
            "flags": {"bit0": "pentagon"},
            "neighbors": "six_uint32_clockwise_missing_is_0xffffffff",
            "dualCorners": "six_triangle_ids_clockwise_missing_is_0xffffffff",
        },
    }
    _atomic_write_json(manifest_path, payload)

    return TopologyCacheInfo(
        directory=directory,
        manifest_path=manifest_path,
        cells_path=cells_path,
        triangles_path=triangles_path,
        cell_crc32=cell_crc,
        triangle_crc32=triangle_crc,
        cell_sha256=cell_sha,
        triangle_sha256=triangle_sha,
    )


def verify_topology_cache(directory: str | Path) -> dict:
    cache_dir = Path(directory)
    manifest_path = cache_dir / "topology.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyError(f"Cannot read topology manifest: {exc}") from exc
    if manifest.get("format") != "HEX_PLANET_TOPOLOGY_CACHE":
        raise TopologyError("Unsupported topology cache format")
    if manifest.get("formatVersion") != 1:
        raise TopologyError("Unsupported topology cache version")

    result = {"valid": True, "files": {}, "issues": []}
    for key in ("cells", "triangles"):
        entry = manifest.get("files", {}).get(key, {})
        path = cache_dir / str(entry.get("path", ""))
        try:
            payload = path.read_bytes()
        except OSError as exc:
            result["valid"] = False
            result["issues"].append(f"cannot read {key}: {exc}")
            continue
        actual_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
        actual_sha = hashlib.sha256(payload).hexdigest()
        expected_crc = str(entry.get("crc32", "")).lower()
        expected_sha = str(entry.get("sha256", "")).lower()
        checksum_valid = actual_crc == expected_crc and actual_sha == expected_sha
        structure_valid = True
        structure_issue = ""
        try:
            if key == "cells":
                if len(payload) < CELL_HEADER.size:
                    raise TopologyError("cells header is truncated")
                magic, version, frequency, cell_count, triangle_count = CELL_HEADER.unpack_from(payload)
                if magic != CELL_MAGIC or version != BINARY_VERSION:
                    raise TopologyError("cells header is unsupported")
                expected_size = CELL_HEADER.size + cell_count * CELL_RECORD.size
            else:
                if len(payload) < TRIANGLE_HEADER.size:
                    raise TopologyError("triangles header is truncated")
                magic, version, frequency, cell_count, triangle_count = TRIANGLE_HEADER.unpack_from(payload)
                if magic != TRIANGLE_MAGIC or version != BINARY_VERSION:
                    raise TopologyError("triangles header is unsupported")
                expected_size = TRIANGLE_HEADER.size + triangle_count * TRIANGLE_RECORD.size
            manifest_frequency = int(manifest.get("topology", {}).get("frequency", -1))
            manifest_cells = int(manifest.get("counts", {}).get("cells", -1))
            manifest_triangles = int(manifest.get("counts", {}).get("triangles", -1))
            if frequency != manifest_frequency or cell_count != manifest_cells or triangle_count != manifest_triangles:
                raise TopologyError(f"{key} header does not match topology.json")
            if len(payload) != expected_size:
                raise TopologyError(f"{key} file size mismatch")
        except (TopologyError, struct.error, ValueError) as exc:
            structure_valid = False
            structure_issue = str(exc)

        valid = checksum_valid and structure_valid
        result["files"][key] = {
            "valid": valid,
            "bytes": len(payload),
            "crc32": actual_crc,
            "sha256": actual_sha,
            "structureValid": structure_valid,
        }
        if not checksum_valid:
            result["valid"] = False
            result["issues"].append(f"checksum mismatch for {key}")
        if not structure_valid:
            result["valid"] = False
            result["issues"].append(structure_issue)
    return result
