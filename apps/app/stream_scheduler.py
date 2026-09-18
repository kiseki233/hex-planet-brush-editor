from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StreamSchedule:
    desired_chunk_ids: tuple[int, ...]
    next_active_chunk_ids: tuple[int, ...]
    add_chunk_ids: tuple[int, ...]
    remove_chunk_ids: tuple[int, ...]
    remaining_additions: int
    remaining_removals: int

    @property
    def complete(self) -> bool:
        return self.remaining_additions == 0 and self.remaining_removals == 0


class VisibleChunkScheduler:
    """Budgeted deterministic transition between visible chunk sets."""

    def __init__(self, max_additions: int = 48, max_removals: int = 96) -> None:
        if max_additions < 1 or max_removals < 1:
            raise ValueError("stream budgets must be positive")
        self.max_additions = int(max_additions)
        self.max_removals = int(max_removals)
        self.generation = 0
        self.desired: tuple[int, ...] = ()

    def set_budgets(self, max_additions: int, max_removals: int) -> None:
        if max_additions < 1 or max_removals < 1:
            raise ValueError("stream budgets must be positive")
        self.max_additions = int(max_additions)
        self.max_removals = int(max_removals)

    def schedule(
        self,
        current: Iterable[int],
        desired: Iterable[int],
        priorities: Mapping[int, float] | None = None,
    ) -> StreamSchedule:
        self.generation += 1
        current_set = set(int(value) for value in current)
        desired_set = set(int(value) for value in desired)
        priorities = priorities or {}
        additions = sorted(
            desired_set - current_set,
            key=lambda item: (float(priorities.get(item, 0.0)), item),
        )
        removals = sorted(
            current_set - desired_set,
            key=lambda item: (-float(priorities.get(item, 0.0)), item),
        )
        selected_remove = tuple(removals[: self.max_removals])
        after_remove = current_set - set(selected_remove)
        selected_add = tuple(additions[: self.max_additions])
        next_active = tuple(sorted(after_remove | set(selected_add)))
        self.desired = tuple(sorted(desired_set))
        return StreamSchedule(
            desired_chunk_ids=self.desired,
            next_active_chunk_ids=next_active,
            add_chunk_ids=selected_add,
            remove_chunk_ids=selected_remove,
            remaining_additions=max(0, len(additions) - len(selected_add)),
            remaining_removals=max(0, len(removals) - len(selected_remove)),
        )
