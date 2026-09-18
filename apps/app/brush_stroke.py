from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .brush_catalog import BrushRecord


class BrushStrokeError(ValueError):
    pass


@dataclass(frozen=True)
class BrushStrokeTool:
    tool: str
    diameter: int
    records: tuple[BrushRecord, ...] = ()

    def validated(self) -> "BrushStrokeTool":
        if self.tool not in {"paint", "erase", "undo"}:
            raise BrushStrokeError(f"Unsupported brush tool: {self.tool}")
        if self.diameter < 1 or self.diameter > 500:
            raise BrushStrokeError("Brush diameter must be between 1 and 500 cells")
        if self.tool == "paint" and not self.records:
            raise BrushStrokeError("The selected brush group has no active images")
        return self


@dataclass
class BrushStrokeState:
    tool: BrushStrokeTool
    last_cell_id: int | None = None
    painted_cell_ids: set[int] = field(default_factory=set)
    touched_cell_count: int = 0
    # Best known graph distance from each reached cell to the stroke path so far.
    # Carrying this across segments is what makes a held stroke incremental: a
    # cell already known to sit at distance d is only re-expanded if the new path
    # segment brings it closer, so a small pointer move expands the new crescent
    # instead of the whole disc.
    distance_to_path: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BrushStrokePlan:
    centerline: tuple[int, ...]
    affected_cell_ids: tuple[int, ...]


class BrushStrokePlanner:
    """Plan continuous graph strokes over the procedural dual topology.

    A stroke is first connected with a topology path. The path is then dilated by
    the brush radius in graph steps. This keeps fast mouse motion continuous and
    avoids treating the stroke as a sequence of unrelated clicks.
    """

    def __init__(self, topology) -> None:
        self.topology = topology
        self._pentagons = frozenset(int(item) for item in topology.pentagon_ids)

    @staticmethod
    def graph_radius(diameter: int) -> int:
        diameter = int(diameter)
        if diameter < 1 or diameter > 500:
            raise BrushStrokeError("Brush diameter must be between 1 and 500 cells")
        return diameter // 2

    def connect(self, start_cell_id: int, end_cell_id: int) -> tuple[int, ...]:
        start = int(start_cell_id)
        end = int(end_cell_id)
        if start == end:
            return (start,)

        target = self.topology.cell_center(end)
        current = start
        path = [current]
        visited = {current}
        maximum_greedy_steps = max(64, int(self._angular_heuristic(start, end) * 3.0) + 32)
        for _ in range(maximum_greedy_steps):
            neighbors = self.topology.cell_neighbors(current)
            if end in neighbors:
                path.append(end)
                return tuple(path)
            best = max(neighbors, key=lambda cell_id: self._dot(self.topology.cell_center(cell_id), target))
            current_score = self._dot(self.topology.cell_center(current), target)
            best_score = self._dot(self.topology.cell_center(best), target)
            if best in visited or best_score <= current_score + 1.0e-15:
                break
            current = best
            visited.add(current)
            path.append(current)

        return self._a_star(start, end)

    def plan_segment(
        self,
        state: BrushStrokeState,
        current_cell_id: int,
    ) -> BrushStrokePlan:
        current_cell_id = int(current_cell_id)
        if state.last_cell_id is None:
            centerline = (current_cell_id,)
        else:
            centerline = self.connect(state.last_cell_id, current_cell_id)
        fresh = self._expand(state, centerline)
        state.last_cell_id = current_cell_id
        state.touched_cell_count += len(fresh)
        return BrushStrokePlan(centerline=centerline, affected_cell_ids=fresh)

    def _expand(
        self, state: BrushStrokeState, centerline: Sequence[int]
    ) -> tuple[int, ...]:
        """Return the cells this segment newly brings within the brush radius.

        The covered region of a stroke is every cell within ``radius`` graph steps
        of any path cell.  Re-dilating the whole disc per segment recomputes that
        entire region on every pointer move and then discards the part already
        painted, which at diameter 500 means expanding 188,000 cells to yield
        1,500 new ones.

        Instead each reached cell keeps its best known distance to the path.  A
        new segment seeds its own path cells at distance zero and relaxes
        outward, expanding a cell only when the new segment strictly improves its
        distance.  The accumulated covered set is identical, because both forms
        compute "distance to the nearest path cell is at most radius"; only the
        redundant re-expansion is gone.
        """
        radius = self.graph_radius(state.tool.diameter)
        distances = state.distance_to_path
        painted = state.painted_cell_ids
        pentagons = self._pentagons
        unreached = radius + 1
        fresh: list[int] = []

        if not distances and len(centerline) == 1:
            fast_disc = getattr(self.topology, "interior_graph_distances", None)
            if fast_disc is not None:
                initial = fast_disc(int(centerline[0]), radius)
                if initial is not None:
                    distances.update(initial)
                    for cell_id in initial:
                        if cell_id in pentagons:
                            continue
                        painted.add(cell_id)
                        fresh.append(cell_id)
                    return tuple(fresh)

        frontier: deque[int] = deque()
        for raw_cell_id in centerline:
            cell_id = int(raw_cell_id)
            if distances.get(cell_id, unreached) == 0:
                continue
            distances[cell_id] = 0
            frontier.append(cell_id)
            if cell_id not in painted and cell_id not in pentagons:
                painted.add(cell_id)
                fresh.append(cell_id)

        if radius == 0:
            return tuple(fresh)

        neighbor_lookup = getattr(
            self.topology, "cell_neighbor_ids_unordered", self.topology.cell_neighbors
        )
        while frontier:
            cell_id = frontier.popleft()
            distance = distances[cell_id]
            if distance >= radius:
                continue
            next_distance = distance + 1
            for neighbor_id in neighbor_lookup(cell_id):
                if distances.get(neighbor_id, unreached) <= next_distance:
                    continue
                distances[neighbor_id] = next_distance
                frontier.append(neighbor_id)
                if neighbor_id not in painted and neighbor_id not in pentagons:
                    painted.add(neighbor_id)
                    fresh.append(neighbor_id)
        return tuple(fresh)

    def _dilate(self, centerline: Sequence[int], diameter: int) -> tuple[int, ...]:
        radius = self.graph_radius(diameter)
        if radius == 0:
            return tuple(dict.fromkeys(int(cell_id) for cell_id in centerline))

        visited = set(int(cell_id) for cell_id in centerline)
        frontier = set(visited)
        neighbor_lookup = getattr(
            self.topology, "cell_neighbor_ids_unordered", self.topology.cell_neighbors
        )
        for _distance in range(radius):
            next_frontier: set[int] = set()
            for cell_id in frontier:
                next_frontier.update(neighbor_lookup(cell_id))
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return tuple(visited)

    def _a_star(self, start: int, end: int) -> tuple[int, ...]:
        frontier: list[tuple[float, int, int]] = []
        sequence = 0
        heapq.heappush(frontier, (self._angular_heuristic(start, end), sequence, start))
        came_from: dict[int, int | None] = {start: None}
        cost: dict[int, int] = {start: 0}
        maximum_expansions = max(4096, int(self._angular_heuristic(start, end) * 48.0) + 2048)
        expansions = 0

        while frontier:
            _priority, _sequence, current = heapq.heappop(frontier)
            if current == end:
                result: list[int] = []
                node: int | None = end
                while node is not None:
                    result.append(node)
                    node = came_from[node]
                result.reverse()
                return tuple(result)
            expansions += 1
            if expansions > maximum_expansions:
                raise BrushStrokeError(
                    f"Cannot connect stroke cells within expansion limit: {start} -> {end}"
                )
            next_cost = cost[current] + 1
            neighbor_lookup = getattr(
                self.topology, "cell_neighbor_ids_unordered", self.topology.cell_neighbors
            )
            for neighbor_id in neighbor_lookup(current):
                if next_cost >= cost.get(neighbor_id, 1 << 60):
                    continue
                cost[neighbor_id] = next_cost
                came_from[neighbor_id] = current
                sequence += 1
                priority = next_cost + self._angular_heuristic(neighbor_id, end)
                heapq.heappush(frontier, (priority, sequence, neighbor_id))
        raise BrushStrokeError(f"Cannot connect stroke cells: {start} -> {end}")

    def _angular_heuristic(self, first: int, second: int) -> float:
        a = self.topology.cell_center(first)
        b = self.topology.cell_center(second)
        angle = math.acos(max(-1.0, min(1.0, self._dot(a, b))))
        # The average adjacent-center angle is close to 1.2/frequency for this grid.
        step = 1.2 / max(1, int(self.topology.frequency))
        return angle / max(step, 1.0e-12)

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def random_assignments(
    cell_ids: Iterable[int],
    tool: BrushStrokeTool,
    *,
    rng: random.Random | None = None,
) -> dict[int, tuple[str, str, int] | None]:
    tool = tool.validated()
    random_source = rng or random.Random()
    assignments: dict[int, tuple[str, str, int] | None] = {}
    if tool.tool == "undo":
        raise BrushStrokeError(
            "The undo brush restores raw snapshot values and does not assign brushes"
        )
    if tool.tool == "erase":
        for cell_id in cell_ids:
            assignments[int(cell_id)] = None
        return assignments

    records = tool.records
    for cell_id in cell_ids:
        record = records[random_source.randrange(len(records))]
        assignments[int(cell_id)] = (
            record.uid,
            record.relative_path,
            random_source.randrange(6),
        )
    return assignments


def records_for_group(
    records_by_uid: Mapping[str, BrushRecord], group_path: str
) -> tuple[BrushRecord, ...]:
    normalized = group_path.strip("/")
    prefix = normalized + "/" if normalized else ""
    records = [
        record
        for record in records_by_uid.values()
        if record.state == "active"
        and (
            record.category_path == normalized
            or (bool(normalized) and record.category_path.startswith(prefix))
            or (not normalized and record.category_path == "")
        )
    ]
    return tuple(sorted(records, key=lambda item: (item.relative_path.casefold(), item.uid)))
