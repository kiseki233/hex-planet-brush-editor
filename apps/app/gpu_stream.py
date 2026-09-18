from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from .brush_catalog import BrushRecord
from .brush_lod import BrushLodCache
from .gpu_batch import (
    EMPTY_LAYER_KEY,
    MISSING_LAYER_KEY,
    GPU_BATCH_VERSION,
    GpuInstance,
    GpuRenderBatch,
    GpuTextureLayer,
    _exact_cell_corners,
    brush_texture_key,
)
from .sphere_map_store import SphereMapSession, SphereMapStore
from .texture_residency import TextureResidencyManager


class GpuStreamError(ValueError):
    pass


@dataclass(frozen=True)
class GpuInstancePatch:
    slot: int
    cell_id: int
    instance: GpuInstance


@dataclass(frozen=True)
class GpuTextureLayerEvent:
    layer: GpuTextureLayer
    replace_existing: bool


@dataclass(frozen=True)
class GpuStreamUpdate:
    active_chunk_ids: tuple[int, ...]
    instance_count: int
    added: tuple[GpuInstancePatch, ...]
    changed: tuple[GpuInstancePatch, ...]
    removed_cell_ids: tuple[int, ...]
    texture_layers: tuple[GpuTextureLayer, ...]
    released_texture_layers: tuple[int, ...] = ()


class VisibleGpuInstanceStream:
    """Dense swap-remove GPU instance stream for current visible chunks.

    Brush texture layers are reference-counted. Callers may either release
    invisible layers after a short grace period or retain them as an LRU cache
    until the configured layer budget is under pressure.
    """

    def __init__(
        self,
        brush_root: str | Path,
        lod_level: int = 0,
        *,
        maximum_texture_layers: int = 4096,
        texture_grace_ticks: int = 2,
        retain_unused_textures: bool = False,
    ) -> None:
        if lod_level < 0 or lod_level > 3:
            raise GpuStreamError("LOD level must be between 0 and 3")
        self.brush_root = Path(brush_root)
        self.lod_level = lod_level
        self.lod_cache = BrushLodCache(self.brush_root)
        self.maximum_texture_layers = int(maximum_texture_layers)
        self.texture_grace_ticks = int(texture_grace_ticks)
        self.retain_unused_textures = bool(retain_unused_textures)
        self.active_chunks: set[int] = set()
        self.chunk_cells: dict[int, tuple[int, ...]] = {}
        self.instances: list[GpuInstance] = []
        self.cell_to_slot: dict[int, int] = {}
        self.layers: list[GpuTextureLayer] = []
        self.texture_events: list[GpuTextureLayerEvent] = []
        self._reset_texture_residency()

    def _reset_texture_residency(self) -> None:
        self.residency = TextureResidencyManager(
            (EMPTY_LAYER_KEY, MISSING_LAYER_KEY),
            maximum_layers=self.maximum_texture_layers,
            grace_ticks=self.texture_grace_ticks,
            retain_unused=self.retain_unused_textures,
        )
        self.layers = [
            GpuTextureLayer(0, EMPTY_LAYER_KEY, None, "", None, "empty"),
            GpuTextureLayer(1, MISSING_LAYER_KEY, None, "", None, "missing"),
        ]
        self.texture_events.clear()

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def layer_by_key(self) -> dict[str, int]:
        return dict(self.residency.key_to_layer)

    def reset_lod(self, lod_level: int) -> None:
        if lod_level < 0 or lod_level > 3:
            raise GpuStreamError("LOD level must be between 0 and 3")
        self.lod_level = int(lod_level)
        self.active_chunks.clear()
        self.chunk_cells.clear()
        self.instances.clear()
        self.cell_to_slot.clear()
        self._reset_texture_residency()

    def set_maximum_texture_layers(self, maximum_texture_layers: int) -> None:
        maximum = max(2, int(maximum_texture_layers))
        highest_layer = max(self.residency.layer_to_key, default=1)
        if highest_layer >= maximum:
            raise GpuStreamError(
                f"Cannot lower texture layer limit below an allocated layer: "
                f"{highest_layer} / {maximum}"
            )
        self.maximum_texture_layers = maximum
        self.residency.maximum_layers = maximum

    def _set_layer(self, layer: int, metadata: GpuTextureLayer) -> None:
        while len(self.layers) <= layer:
            index = len(self.layers)
            self.layers.append(
                GpuTextureLayer(index, f"__free__:{index}", None, "", None, "empty")
            )
        self.layers[layer] = metadata

    def _layer_for_value(
        self,
        session: SphereMapSession,
        local_id: int,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> int:
        if local_id == 0:
            return 0
        entry = session.brush_entries.get(local_id)
        record = None if entry is None else records_by_uid.get(entry.brush_uid)
        if record is None or record.state != "active":
            return 1
        key = brush_texture_key(record, self.lod_level)
        existing = self.residency.layer_for_key(key)
        if existing is not None:
            self.residency.touch_key(key)
            return existing
        allocation = self.residency.allocate(key)
        path = self.lod_cache.existing_path(record, self.lod_level)
        if path is None:
            path = self.lod_cache.ensure(record, self.lod_level).path
        metadata = GpuTextureLayer(
            layer=allocation.layer,
            key=key,
            brush_uid=record.uid,
            content_hash=record.content_hash,
            source_path=path,
            kind="brush",
        )
        self._set_layer(allocation.layer, metadata)
        self.texture_events.append(
            GpuTextureLayerEvent(metadata, allocation.reused)
        )
        return allocation.layer

    def _instance_for_cell(
        self,
        topology,
        session: SphereMapSession,
        value: int,
        cell_id: int,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> GpuInstance:
        local_id = value & 0x0FFF
        rotation = (value >> 12) & 0x0007
        if rotation > 5:
            raise GpuStreamError(f"CellId {cell_id} has invalid rotation {rotation}")
        return GpuInstance(
            cell_id=cell_id,
            corners=_exact_cell_corners(topology, cell_id),
            texture_layer=self._layer_for_value(session, local_id, records_by_uid),
            rotation=rotation,
        )

    def update_visible_chunks(
        self,
        topology,
        layout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        chunk_ids: Iterable[int],
        *,
        worker_pool: Executor | None = None,
    ) -> GpuStreamUpdate:
        requested = set(int(chunk_id) for chunk_id in chunk_ids)
        invalid = [
            chunk_id
            for chunk_id in requested
            if chunk_id < 0 or chunk_id >= layout.chunk_count
        ]
        if invalid:
            raise GpuStreamError(f"ChunkId is outside layout: {invalid[0]}")

        removed_cells: list[int] = []
        for chunk_id in sorted(self.active_chunks - requested):
            cells = self.chunk_cells.pop(chunk_id)
            for cell_id in cells:
                slot = self.cell_to_slot.pop(cell_id)
                last_index = len(self.instances) - 1
                if slot != last_index:
                    moved = self.instances[last_index]
                    self.instances[slot] = moved
                    self.cell_to_slot[moved.cell_id] = slot
                self.instances.pop()
                removed_cells.append(cell_id)
            self.active_chunks.remove(chunk_id)

        added: list[GpuInstancePatch] = []
        pentagons = set(topology.pentagon_ids)
        added_chunk_ids = sorted(requested - self.active_chunks)
        values_by_chunk: dict[int, list[int]] = {}
        missing_chunk_ids: list[int] = []
        for chunk_id in added_chunk_ids:
            values = session.loaded_chunks.get(chunk_id)
            if values is None:
                missing_chunk_ids.append(chunk_id)
            else:
                values_by_chunk[chunk_id] = values
        if worker_pool is not None and len(missing_chunk_ids) > 1:
            futures = {
                chunk_id: worker_pool.submit(
                    store.read_chunk_values, session, chunk_id
                )
                for chunk_id in missing_chunk_ids
            }
            # Merge in deterministic ChunkId order. Worker completion order must
            # never affect dense GPU slots or texture residency.
            for chunk_id in missing_chunk_ids:
                values_by_chunk[chunk_id] = futures[chunk_id].result()
        else:
            for chunk_id in missing_chunk_ids:
                values_by_chunk[chunk_id] = store.read_chunk_values(
                    session, chunk_id
                )

        for chunk_id in added_chunk_ids:
            definition = layout.chunks[chunk_id]
            values = values_by_chunk[chunk_id]
            active_cells: list[int] = []
            for local_index, cell_id in enumerate(definition.cell_ids):
                if cell_id in pentagons:
                    continue
                instance = self._instance_for_cell(
                    topology, session, values[local_index], cell_id, records_by_uid
                )
                slot = len(self.instances)
                self.instances.append(instance)
                self.cell_to_slot[cell_id] = slot
                active_cells.append(cell_id)
                added.append(GpuInstancePatch(slot, cell_id, instance))
            self.chunk_cells[chunk_id] = tuple(active_cells)
            self.active_chunks.add(chunk_id)

        residency_update = self.residency.synchronize(
            instance.texture_layer for instance in self.instances
        )
        for layer in residency_update.released_layers:
            self._set_layer(
                layer,
                GpuTextureLayer(layer, f"__free__:{layer}", None, "", None, "empty"),
            )

        return GpuStreamUpdate(
            active_chunk_ids=tuple(sorted(self.active_chunks)),
            instance_count=len(self.instances),
            added=tuple(added),
            # A consumer performs the same dense swap-removes from
            # removed_cell_ids.  Re-sending the producer's intermediate moved
            # slots is both redundant and invalid after all removals complete.
            changed=(),
            removed_cell_ids=tuple(removed_cells),
            texture_layers=tuple(self.layers),
            released_texture_layers=residency_update.released_layers,
        )

    def patch_cell(
        self,
        topology,
        layout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        cell_id: int,
    ) -> GpuInstancePatch | None:
        slot = self.cell_to_slot.get(cell_id)
        if slot is None:
            return None
        chunk_id, local_index = layout.chunk_for_cell(cell_id)
        values = session.loaded_chunks.get(chunk_id)
        if values is None:
            values = store.read_chunk_values(session, chunk_id)
        instance = self._instance_for_cell(
            topology, session, values[local_index], cell_id, records_by_uid
        )
        self.instances[slot] = instance
        self.residency.synchronize(item.texture_layer for item in self.instances)
        return GpuInstancePatch(slot, cell_id, instance)

    def patch_cells(
        self,
        topology,
        layout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        cell_ids: Iterable[int],
    ) -> tuple[tuple[GpuInstancePatch, ...], tuple[int, ...]]:
        patches: list[GpuInstancePatch] = []
        for raw_cell_id in dict.fromkeys(int(item) for item in cell_ids):
            slot = self.cell_to_slot.get(raw_cell_id)
            if slot is None:
                continue
            chunk_id, local_index = layout.chunk_for_cell(raw_cell_id)
            values = session.loaded_chunks.get(chunk_id)
            if values is None:
                values = store.read_chunk_values(session, chunk_id)
            instance = self._instance_for_cell(
                topology, session, values[local_index], raw_cell_id, records_by_uid
            )
            self.instances[slot] = instance
            patches.append(GpuInstancePatch(slot, raw_cell_id, instance))

        residency_update = self.residency.synchronize(
            item.texture_layer for item in self.instances
        )
        for layer in residency_update.released_layers:
            self._set_layer(
                layer,
                GpuTextureLayer(layer, f"__free__:{layer}", None, "", None, "empty"),
            )
        return tuple(patches), residency_update.released_layers

    def drain_texture_events(self) -> tuple[GpuTextureLayerEvent, ...]:
        events = tuple(self.texture_events)
        self.texture_events.clear()
        return events

    def snapshot_batch(self, topology, layout) -> GpuRenderBatch:
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(topology.stable_hash))
        digest.update(bytes.fromhex(layout.stable_hash))
        digest.update(bytes([self.lod_level]))
        for instance in self.instances:
            digest.update(instance.packed())
            digest.update(instance.cell_id.to_bytes(4, "little", signed=False))
        for layer in self.layers:
            digest.update(layer.key.encode("utf-8"))
        return GpuRenderBatch(
            version=GPU_BATCH_VERSION,
            lod_level=self.lod_level,
            effective_size=self.lod_cache.effective_size(self.lod_level),
            padded_size=self.lod_cache.padded_size(self.lod_level),
            topology_hash=topology.stable_hash,
            layout_hash=layout.stable_hash,
            instances=tuple(self.instances),
            texture_layers=tuple(self.layers),
            stable_hash=digest.hexdigest(),
        )

    def instance_bytes(self) -> bytes:
        return b"".join(instance.packed() for instance in self.instances)


def apply_instance_delta(
    instances: list[GpuInstance],
    cell_to_slot: dict[int, int],
    removed_cell_ids: Iterable[int],
    changed: Iterable[GpuInstancePatch],
    added: Iterable[GpuInstancePatch],
) -> None:
    """Apply the exact swap-remove/add delta emitted by VisibleGpuInstanceStream."""
    for cell_id in removed_cell_ids:
        slot = cell_to_slot.pop(int(cell_id), None)
        if slot is None:
            continue
        last_index = len(instances) - 1
        if slot != last_index:
            moved = instances[last_index]
            instances[slot] = moved
            cell_to_slot[moved.cell_id] = slot
        instances.pop()
    for item in added:
        if item.slot != len(instances):
            raise GpuStreamError(
                f"Added instance slot mismatch: expected {len(instances)}, received {item.slot}"
            )
        instances.append(item.instance)
        cell_to_slot[item.cell_id] = item.slot
