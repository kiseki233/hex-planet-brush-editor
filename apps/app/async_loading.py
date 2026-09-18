from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Iterable

from .sphere_map_store import SphereMapError, SphereMapSession, SphereMapStore


@dataclass(frozen=True)
class AsyncChunkUpdate:
    loaded: tuple[int, ...]
    unloaded: tuple[int, ...]
    retained_dirty: tuple[int, ...]
    discarded: tuple[int, ...]
    errors: tuple[tuple[int, str], ...]
    pending: int


class AsyncViewportChunkLoader:
    def __init__(self, store: SphereMapStore, max_workers: int = 2, max_inflight: int = 8) -> None:
        if max_workers < 1 or max_inflight < 1:
            raise ValueError("max_workers and max_inflight must be positive")
        self.store = store
        self.max_inflight = max_inflight
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hexplanet-pack",
        )
        self.session: SphereMapSession | None = None
        self.required: set[int] = set()
        self.pending: dict[int, concurrent.futures.Future[list[int]]] = {}
        self.failed: dict[int, str] = {}
        self.closed = False

    def set_session(self, session: SphereMapSession | None) -> None:
        if session is self.session:
            return
        self.session = session
        self.required.clear()
        self.failed.clear()
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()

    def request(self, session: SphereMapSession, required_chunk_ids: Iterable[int]) -> AsyncChunkUpdate:
        if self.closed:
            raise RuntimeError("AsyncViewportChunkLoader is closed")
        self.set_session(session)
        required = set(required_chunk_ids)
        invalid = sorted(
            chunk_id for chunk_id in required if chunk_id < 0 or chunk_id >= session.layout.chunk_count
        )
        if invalid:
            raise SphereMapError(f"Viewport requested invalid chunk ids: {invalid[:8]}")
        newly_required = required.difference(self.required)
        for chunk_id in newly_required:
            self.failed.pop(chunk_id, None)
        for chunk_id in tuple(self.failed):
            if chunk_id not in required:
                self.failed.pop(chunk_id, None)
        self.required = required

        unloaded: list[int] = []
        retained_dirty: list[int] = []
        for chunk_id in tuple(session.loaded_chunks):
            if chunk_id in required:
                continue
            if chunk_id in session.dirty_chunks:
                retained_dirty.append(chunk_id)
            else:
                session.loaded_chunks.pop(chunk_id, None)
                unloaded.append(chunk_id)

        self._schedule_available(session)
        return AsyncChunkUpdate(
            loaded=(),
            unloaded=tuple(sorted(unloaded)),
            retained_dirty=tuple(sorted(retained_dirty)),
            discarded=(),
            errors=(),
            pending=len(self.pending),
        )

    def poll(self, session: SphereMapSession) -> AsyncChunkUpdate:
        if session is not self.session:
            self.set_session(session)
        loaded: list[int] = []
        discarded: list[int] = []
        errors: list[tuple[int, str]] = []
        for chunk_id, future in tuple(self.pending.items()):
            if not future.done():
                continue
            self.pending.pop(chunk_id, None)
            try:
                values = future.result()
            except Exception as exc:
                message = str(exc)
                self.failed[chunk_id] = message
                errors.append((chunk_id, message))
                continue
            if chunk_id in session.loaded_chunks:
                discarded.append(chunk_id)
            elif chunk_id in self.required or chunk_id in session.dirty_chunks:
                session.loaded_chunks[chunk_id] = values
                loaded.append(chunk_id)
            else:
                discarded.append(chunk_id)
        self._schedule_available(session)
        return AsyncChunkUpdate(
            loaded=tuple(sorted(loaded)),
            unloaded=(),
            retained_dirty=(),
            discarded=tuple(sorted(discarded)),
            errors=tuple(errors),
            pending=len(self.pending),
        )

    def ensure_loaded(self, session: SphereMapSession, chunk_id: int) -> list[int]:
        if chunk_id in session.loaded_chunks:
            return session.loaded_chunks[chunk_id]
        future = self.pending.pop(chunk_id, None)
        if future is not None:
            try:
                values = future.result()
            except Exception as exc:
                raise SphereMapError(str(exc)) from exc
            session.loaded_chunks[chunk_id] = values
            self._schedule_available(session)
            return values
        values = self.store.load_chunk(session, chunk_id)
        self._schedule_available(session)
        return values

    def _schedule_available(self, session: SphereMapSession) -> None:
        if self.closed or session is not self.session:
            return
        candidates = sorted(
            self.required.difference(session.loaded_chunks).difference(self.pending).difference(self.failed)
        )
        available = self.max_inflight - len(self.pending)
        for chunk_id in candidates[:available]:
            self.pending[chunk_id] = self.executor.submit(self.store.read_chunk_values, session, chunk_id)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)
