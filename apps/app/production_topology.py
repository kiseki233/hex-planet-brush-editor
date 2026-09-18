from __future__ import annotations

import bisect
import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, overload

from .topology import BASE_EDGES, BASE_EDGE_INDEX, BASE_FACES, BASE_VERTICES, TOPOLOGY_TYPE

PRODUCTION_GENERATOR_VERSION = 2
PRODUCTION_ORIENTATION = "default_v1"
CELL_ID_SCHEME = "icosahedral_barycentric_v1_compatible"


class ProductionTopologyError(ValueError):
    pass


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


def _subtract(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length <= 0.0:
        raise ProductionTopologyError("Cannot normalize a zero-length vector")
    return vector[0] / length, vector[1] / length, vector[2] / length


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
    projection = _subtract(point, tuple(center[index] * _dot(point, center) for index in range(3)))
    return math.atan2(_dot(projection, tangent_y), _dot(projection, tangent_x))


_FACE_VERTEX_POSITION = tuple(
    {vertex_id: position for position, vertex_id in enumerate(face)} for face in BASE_FACES
)
_VERTEX_FACES = tuple(
    tuple(face_id for face_id, face in enumerate(BASE_FACES) if vertex_id in face)
    for vertex_id in range(len(BASE_VERTICES))
)
_EDGE_FACES = {
    edge: tuple(face_id for face_id, face in enumerate(BASE_FACES) if edge[0] in face and edge[1] in face)
    for edge in BASE_EDGES
}


@dataclass(frozen=True)
class CellAddress:
    face_id: int
    weight_a: int
    weight_b: int
    weight_c: int

    @property
    def weights(self) -> tuple[int, int, int]:
        return self.weight_a, self.weight_b, self.weight_c


@dataclass(frozen=True)
class ProductionTopologyValidation:
    valid: bool
    frequency: int
    cell_count: int
    triangle_count: int
    edge_count: int
    pentagon_count: int
    checked_cells: int
    reciprocal_neighbors: bool
    issues: tuple[str, ...]


class _CellCenterSequence(Sequence[tuple[float, float, float]]):
    def __init__(self, topology: "ProductionTopology") -> None:
        self.topology = topology

    def __len__(self) -> int:
        return self.topology.cell_count

    @overload
    def __getitem__(self, index: int) -> tuple[float, float, float]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[float, float, float], ...]: ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self.topology.cell_center(cell_id) for cell_id in range(*index.indices(len(self))))
        return self.topology.cell_center(index)


class _NeighborSequence(Sequence[tuple[int, ...]]):
    def __init__(self, topology: "ProductionTopology") -> None:
        self.topology = topology

    def __len__(self) -> int:
        return self.topology.cell_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self.topology.cell_neighbors(cell_id) for cell_id in range(*index.indices(len(self))))
        return self.topology.cell_neighbors(index)


class _IncidentSequence(Sequence[tuple[int, ...]]):
    def __init__(self, topology: "ProductionTopology") -> None:
        self.topology = topology

    def __len__(self) -> int:
        return self.topology.cell_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self.topology.incident_corner_ids(cell_id) for cell_id in range(*index.indices(len(self))))
        return self.topology.incident_corner_ids(index)


class _CornerCenterSequence(Sequence[tuple[float, float, float]]):
    def __init__(self, topology: "ProductionTopology") -> None:
        self.topology = topology

    def __len__(self) -> int:
        return self.topology.cell_count * 6

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cell_id, corner_index = divmod(index, 6)
        corners = self.topology.cell_corners(cell_id)
        if corner_index >= len(corners):
            raise IndexError(index)
        return corners[corner_index]


class ProductionTopology:
    """Procedural icosahedral-dual topology with v1-compatible CellIds.

    Geometry and adjacency are calculated on demand. The object therefore supports
    frequency=1004 without allocating ten million Python tuples or twenty million
    triangle records.
    """

    topology_type = TOPOLOGY_TYPE
    generator_version = PRODUCTION_GENERATOR_VERSION
    orientation = PRODUCTION_ORIENTATION
    cell_id_scheme = CELL_ID_SCHEME
    procedural = True

    def __init__(self, frequency: int) -> None:
        if frequency < 1:
            raise ProductionTopologyError("Frequency must be positive")
        self.frequency = int(frequency)
        self.edge_point_count = max(0, self.frequency - 1)
        self.interior_per_face = max(0, (self.frequency - 1) * (self.frequency - 2) // 2)
        self.edge_start = 12
        self.face_start = self.edge_start + len(BASE_EDGES) * self.edge_point_count
        self._row_starts = tuple(
            (weight_b - 1) * (self.frequency - 1) - ((weight_b - 1) * weight_b) // 2
            for weight_b in range(1, max(1, self.frequency - 1))
        )
        self.stable_hash = self._stable_hash()
        self.pentagon_ids = tuple(range(12))
        self.cell_centers = _CellCenterSequence(self)
        self.neighbors = _NeighborSequence(self)
        self.incident_triangles = _IncidentSequence(self)
        self.triangle_centers = _CornerCenterSequence(self)

    @property
    def cell_count(self) -> int:
        return 10 * self.frequency * self.frequency + 2

    @property
    def triangle_count(self) -> int:
        return 20 * self.frequency * self.frequency

    @property
    def edge_count(self) -> int:
        return 30 * self.frequency * self.frequency

    @property
    def hexagon_count(self) -> int:
        return self.cell_count - 12

    def _stable_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<III", self.generator_version, self.frequency, len(BASE_FACES)))
        digest.update(self.topology_type.encode("ascii"))
        digest.update(self.orientation.encode("ascii"))
        digest.update(self.cell_id_scheme.encode("ascii"))
        digest.update(b"procedural_centers_neighbors_dual_corners_v1")
        return digest.hexdigest()

    def _check_cell_id(self, cell_id: int) -> int:
        cell_id = int(cell_id)
        if cell_id < 0 or cell_id >= self.cell_count:
            raise IndexError(f"CellId is outside topology: {cell_id}")
        return cell_id

    def point_id(self, face_id: int, weight_a: int, weight_b: int, weight_c: int) -> int:
        if face_id < 0 or face_id >= len(BASE_FACES):
            raise ProductionTopologyError(f"Invalid base face: {face_id}")
        if weight_a + weight_b + weight_c != self.frequency:
            raise ProductionTopologyError("Barycentric weights do not match frequency")
        if min(weight_a, weight_b, weight_c) < 0:
            raise ProductionTopologyError("Barycentric weights must not be negative")
        a, b, c = BASE_FACES[face_id]
        if weight_a == self.frequency:
            return a
        if weight_b == self.frequency:
            return b
        if weight_c == self.frequency:
            return c
        if weight_c == 0:
            return self._edge_point_id(a, b, weight_a, weight_b)
        if weight_b == 0:
            return self._edge_point_id(a, c, weight_a, weight_c)
        if weight_a == 0:
            return self._edge_point_id(b, c, weight_b, weight_c)
        rank = (weight_b - 1) * (self.frequency - 1) - ((weight_b - 1) * weight_b) // 2 + (weight_c - 1)
        return self.face_start + face_id * self.interior_per_face + rank

    def _edge_point_id(self, first: int, second: int, first_weight: int, second_weight: int) -> int:
        low, high = sorted((first, second))
        high_weight = second_weight if second == high else first_weight
        if high_weight <= 0 or high_weight >= self.frequency:
            raise ProductionTopologyError("Edge point weight is outside the interior edge range")
        edge_index = BASE_EDGE_INDEX[(low, high)]
        return self.edge_start + edge_index * self.edge_point_count + high_weight - 1

    @lru_cache(maxsize=131072)
    def cell_representations(self, cell_id: int) -> tuple[CellAddress, ...]:
        cell_id = self._check_cell_id(cell_id)
        if cell_id < 12:
            result: list[CellAddress] = []
            for face_id in _VERTEX_FACES[cell_id]:
                weights = [0, 0, 0]
                weights[_FACE_VERTEX_POSITION[face_id][cell_id]] = self.frequency
                result.append(CellAddress(face_id, *weights))
            return tuple(sorted(result, key=lambda item: item.face_id))

        if cell_id < self.face_start:
            if self.edge_point_count <= 0:
                raise ProductionTopologyError("Unexpected edge CellId at frequency 1")
            offset = cell_id - self.edge_start
            edge_index, position_zero = divmod(offset, self.edge_point_count)
            low, high = BASE_EDGES[edge_index]
            high_weight = position_zero + 1
            low_weight = self.frequency - high_weight
            result = []
            for face_id in _EDGE_FACES[(low, high)]:
                weights = [0, 0, 0]
                weights[_FACE_VERTEX_POSITION[face_id][low]] = low_weight
                weights[_FACE_VERTEX_POSITION[face_id][high]] = high_weight
                result.append(CellAddress(face_id, *weights))
            return tuple(sorted(result, key=lambda item: item.face_id))

        if self.interior_per_face <= 0:
            raise ProductionTopologyError("Unexpected interior CellId")
        offset = cell_id - self.face_start
        face_id, rank = divmod(offset, self.interior_per_face)
        row_index = bisect.bisect_right(self._row_starts, rank) - 1
        if row_index < 0:
            raise ProductionTopologyError(f"Cannot decode interior CellId {cell_id}")
        weight_b = row_index + 1
        weight_c = rank - self._row_starts[row_index] + 1
        weight_a = self.frequency - weight_b - weight_c
        if min(weight_a, weight_b, weight_c) <= 0:
            raise ProductionTopologyError(f"Decoded invalid interior CellId {cell_id}")
        return (CellAddress(face_id, weight_a, weight_b, weight_c),)

    def owner_address(self, cell_id: int) -> CellAddress:
        return min(self.cell_representations(cell_id), key=lambda item: item.face_id)

    @lru_cache(maxsize=131072)
    def cell_center(self, cell_id: int) -> tuple[float, float, float]:
        address = self.owner_address(cell_id)
        a_id, b_id, c_id = BASE_FACES[address.face_id]
        a = BASE_VERTICES[a_id]
        b = BASE_VERTICES[b_id]
        c = BASE_VERTICES[c_id]
        return _normalize(
            (
                a[0] * address.weight_a + b[0] * address.weight_b + c[0] * address.weight_c,
                a[1] * address.weight_a + b[1] * address.weight_b + c[1] * address.weight_c,
                a[2] * address.weight_a + b[2] * address.weight_b + c[2] * address.weight_c,
            )
        )

    @lru_cache(maxsize=262144)
    def cell_neighbor_ids_unordered(self, cell_id: int) -> tuple[int, ...]:
        """Return adjacent CellIds without geometric angle sorting.

        Brush expansion and graph search do not need clockwise order. Avoiding
        center calculations makes large graph-radius brushes substantially faster.

        The cache must be able to hold one whole brush footprint. A 500-cell
        brush covers 188,251 cells, so the previous 32,768 entries meant a held
        stroke recomputed every neighbour set from scratch on each pointer move:
        measured at 427 ms per move against 46 ms once the disc fits. The cost is
        about 280 bytes per entry, so this cache tops out near 70 MB, in line with
        the 131,072-entry caches this class already keeps for centers and
        representations.
        """
        cell_id = self._check_cell_id(cell_id)

        # Almost every production cell is strictly inside one base face. Decode
        # its triangular row once and calculate the six neighbouring ranks
        # directly. The generic representation/set/point_id path below is still
        # required at base edges and vertices, but using it for a radius-250
        # brush created hundreds of thousands of temporary objects on its first
        # dab and monopolised the Python GIL for more than a second.
        if cell_id >= self.face_start and self.interior_per_face > 0:
            offset = cell_id - self.face_start
            face_id, rank = divmod(offset, self.interior_per_face)
            row_index = bisect.bisect_right(self._row_starts, rank) - 1
            weight_b = row_index + 1
            weight_c = rank - self._row_starts[row_index] + 1
            weight_a = self.frequency - weight_b - weight_c
            if min(weight_a, weight_b, weight_c) > 1:
                face_offset = self.face_start + face_id * self.interior_per_face

                def interior_id(next_b: int, next_c: int) -> int:
                    next_rank = (
                        (next_b - 1) * (self.frequency - 1)
                        - ((next_b - 1) * next_b) // 2
                        + (next_c - 1)
                    )
                    return face_offset + next_rank

                return tuple(
                    sorted(
                        (
                            interior_id(weight_b - 1, weight_c),
                            interior_id(weight_b, weight_c - 1),
                            interior_id(weight_b + 1, weight_c),
                            interior_id(weight_b + 1, weight_c - 1),
                            interior_id(weight_b, weight_c + 1),
                            interior_id(weight_b - 1, weight_c + 1),
                        )
                    )
                )

        candidates: set[int] = set()
        directions = (
            (1, -1, 0),
            (1, 0, -1),
            (-1, 1, 0),
            (0, 1, -1),
            (-1, 0, 1),
            (0, -1, 1),
        )
        for address in self.cell_representations(cell_id):
            for delta_a, delta_b, delta_c in directions:
                weights = (
                    address.weight_a + delta_a,
                    address.weight_b + delta_b,
                    address.weight_c + delta_c,
                )
                if min(weights) < 0:
                    continue
                neighbor_id = self.point_id(address.face_id, *weights)
                if neighbor_id != cell_id:
                    candidates.add(neighbor_id)
        expected = 5 if cell_id < 12 else 6
        if len(candidates) != expected:
            raise ProductionTopologyError(
                f"CellId {cell_id} has {len(candidates)} procedural neighbors; expected {expected}"
            )
        return tuple(sorted(candidates))

    def interior_graph_distances(
        self, cell_id: int, radius: int
    ) -> dict[int, int] | None:
        """Build an exact hex disc without graph traversal when it stays in one face.

        Barycentric ``(b, c)`` coordinates form an axial hex grid inside a base
        face.  A disc wholly inside that face can therefore be enumerated with
        rank arithmetic instead of discovering six neighbours for every cell.
        ``None`` asks the caller to use the seam-safe graph fallback.
        """
        cell_id = self._check_cell_id(cell_id)
        radius = max(0, int(radius))
        if cell_id < self.face_start or self.interior_per_face <= 0:
            return None
        offset = cell_id - self.face_start
        face_id, rank = divmod(offset, self.interior_per_face)
        row_index = bisect.bisect_right(self._row_starts, rank) - 1
        weight_b = row_index + 1
        weight_c = rank - self._row_starts[row_index] + 1
        weight_a = self.frequency - weight_b - weight_c
        if min(weight_a, weight_b, weight_c) <= radius:
            return None

        face_offset = self.face_start + face_id * self.interior_per_face
        result: dict[int, int] = {}
        for delta_b in range(-radius, radius + 1):
            delta_c_min = max(-radius, -delta_b - radius)
            delta_c_max = min(radius, -delta_b + radius)
            next_b = weight_b + delta_b
            row_base = (
                (next_b - 1) * (self.frequency - 1)
                - ((next_b - 1) * next_b) // 2
            )
            for delta_c in range(delta_c_min, delta_c_max + 1):
                next_c = weight_c + delta_c
                next_id = face_offset + row_base + next_c - 1
                result[next_id] = max(
                    abs(delta_b), abs(delta_c), abs(delta_b + delta_c)
                )
        return result

    @lru_cache(maxsize=131072)
    def cell_neighbors(self, cell_id: int) -> tuple[int, ...]:
        cell_id = self._check_cell_id(cell_id)
        candidates = self.cell_neighbor_ids_unordered(cell_id)
        center = self.cell_center(cell_id)
        tangent_x, tangent_y = _local_basis(center)
        return tuple(
            sorted(
                candidates,
                key=lambda neighbor_id: _angle_around(
                    center, self.cell_center(neighbor_id), tangent_x, tangent_y
                ),
            )
        )

    @lru_cache(maxsize=131072)
    def cell_corners(self, cell_id: int) -> tuple[tuple[float, float, float], ...]:
        center = self.cell_center(cell_id)
        neighbors = self.cell_neighbors(cell_id)
        corners = []
        for index, first_id in enumerate(neighbors):
            second_id = neighbors[(index + 1) % len(neighbors)]
            first = self.cell_center(first_id)
            second = self.cell_center(second_id)
            corners.append(
                _normalize(
                    (
                        center[0] + first[0] + second[0],
                        center[1] + first[1] + second[1],
                        center[2] + first[2] + second[2],
                    )
                )
            )
        return tuple(corners)

    def incident_corner_ids(self, cell_id: int) -> tuple[int, ...]:
        cell_id = self._check_cell_id(cell_id)
        degree = 5 if cell_id < 12 else 6
        return tuple(cell_id * 6 + index for index in range(degree))

    def iter_face_points(self, face_id: int) -> Iterator[tuple[int, int, int, int]]:
        if face_id < 0 or face_id >= len(BASE_FACES):
            raise ProductionTopologyError(f"Invalid base face: {face_id}")
        for weight_b in range(self.frequency + 1):
            for weight_c in range(self.frequency - weight_b + 1):
                weight_a = self.frequency - weight_b - weight_c
                yield weight_a, weight_b, weight_c, self.point_id(
                    face_id, weight_a, weight_b, weight_c
                )

    def validate(
        self,
        cell_ids: Iterable[int] | None = None,
        *,
        full_limit: int = 200_000,
    ) -> ProductionTopologyValidation:
        issues: list[str] = []
        if cell_ids is None:
            if self.cell_count <= full_limit:
                checked = tuple(range(self.cell_count))
            else:
                anchors = list(range(12))
                step = max(1, self.cell_count // 4096)
                anchors.extend(range(12, self.cell_count, step))
                anchors.append(self.cell_count - 1)
                checked = tuple(dict.fromkeys(anchors))
        else:
            checked = tuple(dict.fromkeys(int(value) for value in cell_ids))

        reciprocal = True
        for cell_id in checked:
            try:
                neighbors = self.cell_neighbors(cell_id)
            except (IndexError, ProductionTopologyError) as exc:
                issues.append(str(exc))
                continue
            if len(neighbors) != len(set(neighbors)):
                issues.append(f"CellId {cell_id} has duplicate neighbors")
            for neighbor_id in neighbors:
                if cell_id not in self.cell_neighbors(neighbor_id):
                    reciprocal = False
                    issues.append(f"Neighbor relation is not reciprocal: {cell_id}->{neighbor_id}")
                    break

        return ProductionTopologyValidation(
            valid=not issues,
            frequency=self.frequency,
            cell_count=self.cell_count,
            triangle_count=self.triangle_count,
            edge_count=self.edge_count,
            pentagon_count=12,
            checked_cells=len(checked),
            reciprocal_neighbors=reciprocal,
            issues=tuple(issues),
        )
