from __future__ import annotations

import array
from dataclasses import dataclass, field
from typing import Sequence

# Cell states are uint16.  Snapshots are kept as packed buffers rather than
# Python lists: a 258-cell chunk costs 516 bytes packed against roughly 9 KB as a
# tuple of boxed integers, which is the difference between a 500-cell stroke
# costing 376 KB and costing 6.8 MB.
CELL_TYPECODE = "H"
DEFAULT_DEPTH = 32
DEFAULT_BYTE_BUDGET = 192 * 1024 * 1024


class UndoHistoryError(RuntimeError):
    pass


def pack_values(values: Sequence[int]) -> bytes:
    return array.array(CELL_TYPECODE, values).tobytes()


def unpack_values(blob: bytes) -> list[int]:
    buffer = array.array(CELL_TYPECODE)
    buffer.frombytes(blob)
    return buffer.tolist()


@dataclass
class UndoRecord:
    label: str
    chunks: dict[int, bytes] = field(default_factory=dict)
    cell_count: int = 0

    @property
    def byte_size(self) -> int:
        return sum(len(blob) for blob in self.chunks.values())


@dataclass(frozen=True)
class UndoResult:
    label: str
    changed_cell_ids: tuple[int, ...]
    chunk_count: int


class UndoHistory:
    """Per-stroke undo over chunk snapshots.

    A stroke is the unit of undo.  Before a stroke first writes into a chunk the
    chunk's whole cell array is copied, so undo is a plain buffer restore that
    reuses the existing dirty-chunk save path and needs no new on-disk format.

    Redo entries are produced when undoing rather than stored up front, so a
    stroke that is never undone costs one snapshot instead of two.

    Local brush ids are deliberately never recycled by an undo: the snapshot
    holds raw uint16 values that reference entries in the map brush table, and
    other cells may still reference the same entry.  Leaving an orphaned brush
    table entry behind is safe; freeing one would silently corrupt the map.
    """

    def __init__(
        self, depth: int = DEFAULT_DEPTH, byte_budget: int = DEFAULT_BYTE_BUDGET
    ) -> None:
        if depth < 1:
            raise UndoHistoryError("depth must be positive")
        self.depth = int(depth)
        self.byte_budget = int(byte_budget)
        self._undo: list[UndoRecord] = []
        self._redo: list[UndoRecord] = []
        self._pending: UndoRecord | None = None

    # -- capture -------------------------------------------------------------

    def begin(self, label: str) -> None:
        """Open a pending record.  A stroke that is already open is kept."""
        if self._pending is None:
            self._pending = UndoRecord(label=str(label))

    @property
    def capturing(self) -> bool:
        return self._pending is not None

    def has_chunk(self, chunk_id: int) -> bool:
        """Whether the open stroke already snapshotted this chunk."""
        pending = self._pending
        return pending is not None and int(chunk_id) in pending.chunks

    def capture(self, chunk_id: int, values: Sequence[int]) -> None:
        """Snapshot a chunk before its first modification within this stroke."""
        pending = self._pending
        if pending is None:
            return
        key = int(chunk_id)
        if key in pending.chunks:
            return
        pending.chunks[key] = pack_values(values)

    def commit(self, cell_count: int = 0) -> bool:
        pending = self._pending
        self._pending = None
        if pending is None or not pending.chunks:
            return False
        pending.cell_count = int(cell_count)
        self._undo.append(pending)
        self._redo.clear()
        self._trim()
        return True

    def abort(self) -> None:
        self._pending = None

    def _trim(self) -> None:
        while len(self._undo) > self.depth:
            self._undo.pop(0)
        while len(self._undo) > 1 and self.byte_size > self.byte_budget:
            self._undo.pop(0)

    # -- inspection ----------------------------------------------------------

    @property
    def byte_size(self) -> int:
        return sum(record.byte_size for record in self._undo) + sum(
            record.byte_size for record in self._redo
        )

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def redo_depth(self) -> int:
        return len(self._redo)

    def next_undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    def next_redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._pending = None

    def previous_value(self, chunk_id: int, local_index: int) -> int | None:
        """Value a cell held before the most recent stroke that changed it.

        Used by the undo brush.  Walking the stack top-down means a cell that the
        newest stroke never touched still reverts by exactly one change of its
        own, rather than reverting to whatever the newest stroke happened to hit.
        """
        key = int(chunk_id)
        index = int(local_index)
        for record in reversed(self._undo):
            blob = record.chunks.get(key)
            if blob is None:
                continue
            offset = index * 2
            if offset + 2 > len(blob):
                return None
            return int.from_bytes(blob[offset : offset + 2], "little", signed=False)
        return None

    # -- undo / redo ---------------------------------------------------------

    def _restore(
        self, record: UndoRecord, session, store, layout
    ) -> tuple[UndoRecord, tuple[int, ...]]:
        inverse = UndoRecord(label=record.label)
        changed: list[int] = []
        for chunk_id, blob in record.chunks.items():
            values = store.load_chunk(session, chunk_id)
            inverse.chunks[chunk_id] = pack_values(values)
            restored = unpack_values(blob)
            if len(restored) != len(values):
                raise UndoHistoryError(
                    f"Chunk {chunk_id} snapshot holds {len(restored)} cells "
                    f"but the chunk has {len(values)}"
                )
            cell_ids = layout.chunk_cell_ids(int(chunk_id))
            chunk_changed = False
            for local_index, (current, previous) in enumerate(zip(values, restored)):
                if current == previous:
                    continue
                changed.append(int(cell_ids[local_index]))
                chunk_changed = True
            if chunk_changed:
                values[:] = restored
                session.dirty_chunks.add(int(chunk_id))
        inverse.cell_count = len(changed)
        return inverse, tuple(changed)

    def undo(self, session, store, layout) -> UndoResult | None:
        if self._pending is not None:
            raise UndoHistoryError("Cannot undo while a stroke is still being captured")
        if not self._undo:
            return None
        record = self._undo.pop()
        inverse, changed = self._restore(record, session, store, layout)
        self._redo.append(inverse)
        return UndoResult(record.label, changed, len(record.chunks))

    def redo(self, session, store, layout) -> UndoResult | None:
        if self._pending is not None:
            raise UndoHistoryError("Cannot redo while a stroke is still being captured")
        if not self._redo:
            return None
        record = self._redo.pop()
        inverse, changed = self._restore(record, session, store, layout)
        self._undo.append(inverse)
        self._trim()
        return UndoResult(record.label, changed, len(record.chunks))
