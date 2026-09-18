from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from .aggregate_lod import (
    AggregateBuildReport,
    AggregateLodCache,
    AggregateSelection,
    ProductionAggregateHierarchy,
)
from .brush_catalog import BrushRecord
from .gpu_batch import GpuRenderBatch, build_brush_texture_payload
from .gpu_memory import (
    DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
    catalog_texture_layer_limit,
)
from .gpu_edit import (
    GpuBatchResetPatch,
    GpuStreamPatch,
    GpuTextureUpload,
)
from .gpu_stream import GpuStreamUpdate, VisibleGpuInstanceStream
from .production_layout import ProductionChunkLayout
from .production_lod import ProductionLodController, ProductionLodDecision
from .production_topology import ProductionTopology
from .production_visibility import (
    ProductionVisibilityIndex,
    ProductionVisibilityQuery,
)
from .sphere_map_store import SphereMapSession, SphereMapStore
from .stream_scheduler import StreamSchedule, VisibleChunkScheduler
from .zoom_tiers import visible_cells_estimate
from .i18n import t


class ProductionStreamingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductionStreamFrame:
    query: ProductionVisibilityQuery
    update: GpuStreamUpdate
    texture_uploads: tuple[GpuTextureUpload, ...]
    lod: ProductionLodDecision
    schedule: StreamSchedule | None = None
    aggregate_selection: AggregateSelection | None = None
    reset_batch: GpuRenderBatch | None = None

    @property
    def has_more(self) -> bool:
        return self.schedule is not None and not self.schedule.complete

    @property
    def total_chunk_count(self) -> int:
        if self.lod.level >= 4:
            return 0
        desired = (
            self.query.chunk_ids
            if self.schedule is None
            else self.schedule.desired_chunk_ids
        )
        return len(desired)

    @property
    def loaded_chunk_count(self) -> int:
        if self.lod.level >= 4:
            return 0
        desired = set(
            self.query.chunk_ids
            if self.schedule is None
            else self.schedule.desired_chunk_ids
        )
        return len(desired.intersection(self.update.active_chunk_ids))

    @property
    def remaining_chunk_count(self) -> int:
        if self.schedule is None:
            return 0
        return self.schedule.remaining_additions + self.schedule.remaining_removals

    @property
    def load_percent(self) -> int:
        total = self.total_chunk_count
        if total <= 0:
            return 100
        return max(0, min(100, round(self.loaded_chunk_count * 100 / total)))


class ProductionGpuStreamingController:
    """Thread-safe production map owner for visible detail and aggregate streams."""

    def __init__(
        self,
        topology: ProductionTopology,
        layout: ProductionChunkLayout,
        visibility: ProductionVisibilityIndex,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        brush_root,
        *,
        lod_level: int = 0,
        automatic_lod: bool = False,
        # The L3 tier reaches down to zoom 4.5, and its hysteresis edge needs
        # 1,727 chunks on a 1180x720 viewport - about 34 MB of instance data,
        # which the streaming budget adds progressively. The old 1,200 was the
        # reason per-cell rendering could not follow the tier ladder that far out.
        maximum_visible_chunks: int = 2048,
        stream_add_budget: int | None = None,
        stream_remove_budget: int | None = None,
        maximum_texture_layers: int = 4096,
        texture_memory_budget_bytes: int = DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
        parallel_executor: concurrent.futures.Executor | None = None,
        parallel_workers: int | None = None,
    ) -> None:
        if layout.topology_hash != topology.stable_hash:
            raise ProductionStreamingError("Production topology and layout hashes do not match")
        if visibility.layout_hash != layout.stable_hash:
            raise ProductionStreamingError("Production visibility and layout hashes do not match")
        if session.layout.stable_hash != layout.stable_hash:
            raise ProductionStreamingError("Production map and layout hashes do not match")
        self.topology = topology
        self.layout = layout
        self.visibility = visibility
        self.session = session
        self.store = store
        self.records_by_uid = dict(records_by_uid)
        self.parallel_workers = max(
            1,
            int(
                parallel_workers
                if parallel_workers is not None
                else 12
            ),
        )
        self._parallel_executor = parallel_executor
        self._owns_parallel_executor = parallel_executor is None
        self.configured_maximum_texture_layers = max(2, int(maximum_texture_layers))
        self.texture_memory_budget_bytes = max(1, int(texture_memory_budget_bytes))
        active_brush_count = sum(
            1 for record in self.records_by_uid.values() if record.state == "active"
        )
        self.texture_layer_limit = catalog_texture_layer_limit(
            active_brush_count,
            padded_size=520,
            budget_bytes=self.texture_memory_budget_bytes,
            configured_limit=maximum_texture_layers,
        )
        self.pending_hardware_texture_layer_limit: int | None = None
        if lod_level < 0 or lod_level > 4:
            raise ProductionStreamingError("lod_level must be between 0 and 4")
        self.stream = VisibleGpuInstanceStream(
            brush_root,
            lod_level=min(lod_level, 3),
            maximum_texture_layers=self.texture_layer_limit,
            retain_unused_textures=True,
        )
        self.maximum_visible_chunks = max(1, int(maximum_visible_chunks))
        self.automatic_lod = bool(automatic_lod)
        self.lod_controller = ProductionLodController(initial_level=lod_level)
        self.scheduler = (
            VisibleChunkScheduler(
                max_additions=max(1, int(stream_add_budget)),
                max_removals=max(1, int(stream_remove_budget or stream_add_budget * 2)),
            )
            if stream_add_budget is not None
            else None
        )
        self.stream_add_budget = (
            None if stream_add_budget is None else max(1, int(stream_add_budget))
        )
        self.stream_remove_budget = (
            None
            if stream_add_budget is None
            else max(1, int(stream_remove_budget or stream_add_budget * 2))
        )
        self.stream_add_cost_ms = 8.0
        self.aggregate_hierarchy = ProductionAggregateHierarchy(visibility, layout)
        self.aggregate_cache = AggregateLodCache(brush_root)
        self.last_aggregate_selection: AggregateSelection | None = None
        self.last_reset_batch: GpuRenderBatch | None = None
        # The aggregate renderer deliberately reuses stream LOD3 internally,
        # so ``stream.lod_level`` alone cannot tell whether the consumer
        # currently owns aggregate proxy instances or detailed cell instances.
        self.last_output_mode: str | None = None
        # Set when the brush catalog changed under a live stream, so the next
        # frame emits a full reset batch even though the LOD level is unchanged.
        self._force_reset = False
        # A superseded view may mutate the producer stream without sending its
        # serial delta to the GPU. The next stable view then sends one complete
        # batch without discarding and rebuilding the producer stream again.
        self._force_consumer_reset = False
        # Visibility and aggregate selection depend only on the camera, never on
        # map content, so an unchanged camera can reuse both. This matters because
        # filling in a budgeted far view repeats the identical view many times,
        # and recomputing the query and the selection each time costs far more
        # than the handful of nodes that pass actually builds.
        self._view_key: tuple[object, ...] | None = None
        self._view_query: ProductionVisibilityQuery | None = None
        self._view_selection: AggregateSelection | None = None
        self.lock = threading.RLock()

    @property
    def lod_level(self) -> int:
        return self.stream.lod_level

    def request_consumer_reset(self) -> None:
        self._force_consumer_reset = True

    def update_records(self, records_by_uid: Mapping[str, BrushRecord]) -> bool:
        """Adopt a rescanned brush catalog, re-texturing the stream if needed.

        Swapping the dictionary alone is not enough. A texture layer is keyed by
        uid *and* content hash, and instances hold a resolved layer index, so a
        brush replaced in place kept rendering its previous artwork until the
        chunk happened to leave and re-enter the stream - and the superseded
        layer was never released. Detecting a content or availability change and
        resetting the stream forces every visible cell to re-resolve.

        Returns whether the stream was reset.
        """
        with self.lock:
            previous = self.records_by_uid
            self.records_by_uid = dict(records_by_uid)
            if not self._brush_content_changed(previous, self.records_by_uid):
                return False
            self.stream.reset_lod(self.stream.lod_level)
            self._force_reset = True
            self.last_reset_batch = None
            return True

    @staticmethod
    def _brush_content_changed(
        previous: Mapping[str, BrushRecord], current: Mapping[str, BrushRecord]
    ) -> bool:
        for uid, record in current.items():
            before = previous.get(uid)
            if before is None:
                continue
            if before.content_hash != record.content_hash or before.state != record.state:
                return True
        return any(uid not in current for uid in previous)

    def apply_hardware_texture_layer_limit(self, hardware_limit: int) -> int:
        """Keep the producer inside the depth accepted by the live GL context."""

        with self.lock:
            maximum = min(
                self.texture_layer_limit,
                max(2, int(hardware_limit)),
            )
            self.stream.set_maximum_texture_layers(maximum)
            self.texture_layer_limit = maximum
            return maximum

    def request_hardware_texture_layer_limit(self, hardware_limit: int) -> int:
        """Queue a GL limit without making the Tk/UI thread wait on streaming."""

        maximum = min(self.texture_layer_limit, max(2, int(hardware_limit)))
        self.pending_hardware_texture_layer_limit = maximum
        return maximum

    def update_view(
        self,
        yaw: float,
        pitch: float,
        zoom: float,
        width: int,
        height: int,
        *,
        interactive: bool = False,
    ) -> ProductionStreamFrame:
        view_key = (
            round(float(yaw), 6),
            round(float(pitch), 6),
            round(float(zoom), 6),
            int(width),
            int(height),
        )
        if view_key == self._view_key and self._view_query is not None:
            query = self._view_query
        else:
            query = self.visibility.query(self.layout, yaw, pitch, zoom, width, height)
            self._view_key = view_key
            self._view_query = query
            self._view_selection = None
        if self.automatic_lod:
            lod = self.lod_controller.update(zoom, width, height)
        else:
            lod = ProductionLodDecision(
                level=self.stream.lod_level,
                mode="detail",
                candidate_cells=query.candidate_cells,
                changed=False,
                description=t("固定 LOD{lod_level}", lod_level=self.stream.lod_level),
            )

        with self.lock:
            pending_limit = self.pending_hardware_texture_layer_limit
            if pending_limit is not None:
                self.stream.set_maximum_texture_layers(pending_limit)
                self.texture_layer_limit = pending_limit
                self.pending_hardware_texture_layer_limit = None
            if lod.level < 4 and len(query.chunk_ids) > self.maximum_visible_chunks:
                # The tier ladder is calibrated for a typical window. A much
                # larger viewport sees proportionally more chunks at the same
                # zoom, so the per-cell tiers can genuinely exceed the streaming
                # budget. Fall back to the far-view surface for this frame rather
                # than raising: an exception here only reaches a status line, and
                # the viewport would silently stop updating.
                lod = ProductionLodDecision(
                    level=4,
                    mode="aggregate",
                    candidate_cells=lod.candidate_cells,
                    changed=self.lod_controller.level != 4,
                    description=(
                        t("{description}｜候选区块 {len:,} 超出 {maximum_visible_chunks:,}，暂用远景地表", description=lod.description, len=len(query.chunk_ids), maximum_visible_chunks=self.maximum_visible_chunks)
                    ),
                    tier=lod.tier,
                )
                self.lod_controller.force_level(4)
            if lod.level >= 4:
                update = self.stream.update_visible_chunks(
                    self.topology,
                    self.layout,
                    self.session,
                    self.store,
                    self.records_by_uid,
                    (),
                )
                selection = self._view_selection
                if selection is None:
                    selection = self.aggregate_hierarchy.select(
                        yaw, pitch, zoom, width, height, target_pixels=96.0
                    )
                    self._view_selection = selection
                # The far view renders as the continuous inset sphere sampling the
                # saved 1024x512 surface texture - the same decision the software
                # viewport made in v1.0.3. The per-node proxy hexagons this branch
                # used to emit were strictly worse than the surface they covered:
                # one flat average color per up-to-16k-cell node against roughly
                # 5x5 cells per surface texel, and a node whose cache entry did
                # not exist yet rendered as an opaque dark slab. The selection is
                # still computed for statistics and acceptance; only the mosaic
                # is gone, so no aggregate node needs to be generated per view.
                batch = self.stream.snapshot_batch(self.topology, self.layout)
                self.last_aggregate_selection = selection
                self.last_reset_batch = batch
                self.last_output_mode = "aggregate"
                self._release_inactive_clean_chunks()
                return ProductionStreamFrame(
                    query=query,
                    update=update,
                    texture_uploads=(),
                    lod=lod,
                    aggregate_selection=selection,
                    reset_batch=batch,
                )

            leaving_aggregate = self.last_output_mode == "aggregate"
            lod_changed = self.stream.lod_level != lod.level
            producer_reset_required = leaving_aggregate or lod_changed or self._force_reset
            consumer_reset_required = self._force_consumer_reset
            self._force_reset = False
            self._force_consumer_reset = False
            if producer_reset_required:
                self.stream.reset_lod(lod.level)
            reset_required = producer_reset_required or consumer_reset_required

            schedule = None
            requested = query.chunk_ids
            if self.scheduler is not None:
                self._configure_stream_budget(lod.level, bool(interactive))
                priorities = self._chunk_priorities(query.chunk_ids, yaw, pitch)
                schedule = self.scheduler.schedule(
                    self.stream.active_chunks, query.chunk_ids, priorities
                )
                requested = schedule.next_active_chunk_ids
            stream_started = time.perf_counter()
            update = self.stream.update_visible_chunks(
                self.topology,
                self.layout,
                self.session,
                self.store,
                self.records_by_uid,
                requested,
                worker_pool=self._parallel_pool(),
            )
            if schedule is not None and schedule.add_chunk_ids:
                elapsed_ms = (time.perf_counter() - stream_started) * 1000.0
                effective_chunks = (
                    len(schedule.add_chunk_ids)
                    + len(schedule.remove_chunk_ids) * 0.25
                )
                sample = elapsed_ms / max(1.0, effective_chunks)
                self.stream_add_cost_ms = (
                    self.stream_add_cost_ms * 0.75 + sample * 0.25
                )
            uploads = self._drain_texture_uploads()
            self._release_inactive_clean_chunks()
            reset_batch = (
                self.stream.snapshot_batch(self.topology, self.layout)
                if reset_required
                else None
            )
            self.last_reset_batch = reset_batch
            self.last_aggregate_selection = None
            self.last_output_mode = "detail"
            return ProductionStreamFrame(
                query=query,
                update=update,
                texture_uploads=uploads,
                lod=lod,
                schedule=schedule,
                reset_batch=reset_batch,
            )

    def _configure_stream_budget(self, lod_level: int, interactive: bool) -> None:
        scheduler = self.scheduler
        maximum_additions = self.stream_add_budget
        maximum_removals = self.stream_remove_budget
        if (
            scheduler is None
            or maximum_additions is None
            or maximum_removals is None
        ):
            return
        # LOD3 chunks are much more expensive to materialize than near-view
        # chunks. Keep a drag pass short enough that a newer camera can supersede
        # it quickly, then fill aggressively once the mouse is released.
        target_ms = 55.0 if interactive else 600.0
        estimated = max(0.25, float(self.stream_add_cost_ms))
        additions = max(2, min(maximum_additions, int(target_ms / estimated)))
        if lod_level <= 1:
            additions = min(maximum_additions, max(additions, 16))
        elif lod_level == 2:
            additions = min(maximum_additions, max(additions, 8))
        removals = min(
            maximum_removals,
            max(additions * (4 if interactive else 2), additions),
        )
        scheduler.set_budgets(additions, removals)

    def initial_batch(self) -> GpuRenderBatch:
        with self.lock:
            if self.last_reset_batch is not None:
                return self.last_reset_batch
            return self.stream.snapshot_batch(self.topology, self.layout)

    def patch_for_frame(
        self, request_id: int, frame: ProductionStreamFrame
    ) -> GpuStreamPatch | GpuBatchResetPatch:
        if frame.reset_batch is not None:
            if frame.lod.level >= 4:
                selection_count = 0 if frame.aggregate_selection is None else len(frame.aggregate_selection.node_ids)
                message = (
                    t("远景 LOD4：显示实时地表纹理（覆盖 {selection_count:,} 个层级节点范围）；可直接绘制，未保存笔划会局部刷新", selection_count=selection_count)
                )
                # The far view is the saved-surface sphere. Painting there is
                # legitimate - picking resolves an exact CellId regardless of
                # what is rendered - so the viewport must not lock the brush.
                editable = True
            else:
                message = (
                    t("切换 {description}：区块 {len:,}，实例 {instance_count:,}；加载 {loaded_chunk_count:,}/{total_chunk_count:,}（{load_percent}%）", description=frame.lod.description, len=len(frame.update.active_chunk_ids), instance_count=frame.update.instance_count, loaded_chunk_count=frame.loaded_chunk_count, total_chunk_count=frame.total_chunk_count, load_percent=frame.load_percent)
                )
                editable = True
            return GpuBatchResetPatch(
                request_id=request_id,
                batch=frame.reset_batch,
                message=message,
                editable=editable,
                has_more=frame.has_more,
                loaded_chunk_count=frame.loaded_chunk_count,
                total_chunk_count=frame.total_chunk_count,
                remaining_chunk_count=frame.remaining_chunk_count,
                texture_uploads=frame.texture_uploads,
            )
        update = frame.update
        return GpuStreamPatch(
            request_id=request_id,
            active_chunk_ids=update.active_chunk_ids,
            instance_count=update.instance_count,
            removed_cell_ids=update.removed_cell_ids,
            added=update.added,
            changed=update.changed,
            texture_uploads=frame.texture_uploads,
            message=(
                t("{description}：区块 {len:,}，实例 {instance_count:,}，候选格子 {candidate_cells:,}；加载 {loaded_chunk_count:,}/{total_chunk_count:,}（{load_percent}%）", description=frame.lod.description, len=len(update.active_chunk_ids), instance_count=update.instance_count, candidate_cells=frame.query.candidate_cells, loaded_chunk_count=frame.loaded_chunk_count, total_chunk_count=frame.total_chunk_count, load_percent=frame.load_percent)
                + (t("，继续分批加载") if frame.has_more else "")
            ),
            lod_level=frame.lod.level,
            padded_size=self.stream.lod_cache.padded_size(frame.lod.level),
            released_texture_layers=update.released_texture_layers,
            has_more=frame.has_more,
            editable=True,
            loaded_chunk_count=frame.loaded_chunk_count,
            total_chunk_count=frame.total_chunk_count,
            remaining_chunk_count=frame.remaining_chunk_count,
        )

    def patch_visible_cell(self, cell_id: int):
        with self.lock:
            if self.lod_controller.level >= 4 and self.automatic_lod:
                return None, ()
            patch = self.stream.patch_cell(
                self.topology,
                self.layout,
                self.session,
                self.store,
                self.records_by_uid,
                cell_id,
            )
            uploads = self._drain_texture_uploads()
            return patch, uploads

    def patch_visible_cells(self, cell_ids):
        with self.lock:
            if self.lod_controller.level >= 4 and self.automatic_lod:
                return (), (), ()
            patches, released_layers = self.stream.patch_cells(
                self.topology,
                self.layout,
                self.session,
                self.store,
                self.records_by_uid,
                cell_ids,
            )
            uploads = self._drain_texture_uploads()
            return patches, uploads, released_layers

    def save(
        self,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> int:
        with self.lock:
            dirty_ids = tuple(sorted(self.session.dirty_chunks))
            dirty = len(dirty_ids)
            self.store.save(
                self.session,
                progress=progress,
                worker_pool=self._parallel_pool(),
            )
            if dirty_ids:
                self.aggregate_cache.invalidate_chunks(
                    self.aggregate_hierarchy, self.session.map_dir, dirty_ids
                )
            self._release_inactive_clean_chunks()
            return dirty

    def build_aggregate_cache(self) -> AggregateBuildReport:
        with self.lock:
            if self.session.dirty_chunks or self.session.brush_table_dirty:
                raise ProductionStreamingError(t("请先保存地图，再构建聚合远景缓存"))
            return self.aggregate_cache.build_all(
                self.aggregate_hierarchy,
                self.layout,
                self.session,
                self.store,
                self.records_by_uid,
            )

    def _release_inactive_clean_chunks(self) -> None:
        active = self.stream.active_chunks
        for chunk_id in tuple(self.session.loaded_chunks):
            if chunk_id not in active and chunk_id not in self.session.dirty_chunks:
                self.session.loaded_chunks.pop(chunk_id, None)

    def _drain_texture_uploads(self) -> tuple[GpuTextureUpload, ...]:
        jobs: list[tuple[object, BrushRecord]] = []
        for event in self.stream.drain_texture_events():
            layer = event.layer
            if layer.kind != "brush" or layer.brush_uid is None:
                continue
            record = self.records_by_uid.get(layer.brush_uid)
            if record is None or record.state != "active":
                continue
            jobs.append((event, record))

        def build(job):
            event, record = job
            return (
                event,
                record,
                build_brush_texture_payload(
                    self.stream.brush_root,
                    record,
                    self.stream.lod_level,
                ),
            )

        if len(jobs) > 1:
            completed = self._parallel_pool().map(build, jobs)
        else:
            completed = map(build, jobs)
        uploads: list[GpuTextureUpload] = []
        for event, record, payload in completed:
            layer = event.layer
            uploads.append(
                GpuTextureUpload(
                    layer=layer.layer,
                    key=layer.key,
                    width=payload.width,
                    height=payload.height,
                    pixels_rgba=payload.pixels_rgba,
                    replace_existing=event.replace_existing,
                    source_path=str(
                        self.stream.lod_cache.path_for(record, self.stream.lod_level)
                    ),
                )
            )
        return tuple(uploads)

    def _parallel_pool(self) -> concurrent.futures.Executor:
        executor = self._parallel_executor
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="hexplanet-stream",
            )
            self._parallel_executor = executor
        return executor

    def close(self) -> None:
        executor = self._parallel_executor
        self._parallel_executor = None
        if self._owns_parallel_executor and executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _estimated_visible_cells(self, zoom: float, width: int, height: int) -> int:
        """Cells the viewport covers, from the shared spherical-cap estimate.

        This used to be its own ``hexagon_count / (2 * zoom^2)`` approximation,
        which ran about 1.74x higher than the figure the editor displays. Both
        now come from ``zoom_tiers`` so the LOD ladder and the interface cannot
        drift apart again.
        """
        return visible_cells_estimate(
            zoom, width, height, cell_count=self.topology.hexagon_count
        )

    def _chunk_priorities(
        self, chunk_ids: tuple[int, ...], yaw: float, pitch: float
    ) -> dict[int, float]:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        priorities: dict[int, float] = {}
        for chunk_id in chunk_ids:
            x, y, z = self.visibility.chunks[chunk_id].center
            x, z = x * cy + z * sy, -x * sy + z * cy
            y, z = y * cp - z * sp, y * sp + z * cp
            priorities[chunk_id] = x * x + y * y - z * 0.05
        return priorities
