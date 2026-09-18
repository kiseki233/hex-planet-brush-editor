from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .brush_catalog import BrushRecord
from .brush_lod import BrushLodCache
from .distant_lod import DistantLodCache
from .gpu_batch import (
    EMPTY_LAYER_KEY,
    MISSING_LAYER_KEY,
    GPU_BATCH_VERSION,
    GpuInstance,
    GpuRenderBatch,
    GpuTextureLayer,
)
from .png_pixels import PixelImage, atomic_write_png
from .production_layout import ProductionChunkLayout
from .production_visibility import (
    ProductionVisibilityIndex,
    _cap_intersects_viewport,
    _cap_from_bounds,
    _widest_axis,
)
from .sphere_map_store import SphereMapSession, SphereMapStore

AGGREGATE_CACHE_VERSION = 1
AGGREGATE_SIDE = 72
AGGREGATE_EFFECTIVE = 64
AGGREGATE_CELL_BASE = 0x80000000


class AggregateLodError(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateNode:
    node_id: int
    center: tuple[float, float, float]
    angular_radius: float
    left: int
    right: int
    chunk_ids: tuple[int, ...]

    @property
    def is_leaf(self) -> bool:
        return self.left < 0 and self.right < 0


@dataclass(frozen=True)
class AggregateSelection:
    node_ids: tuple[int, ...]
    visited_nodes: int
    rejected_nodes: int
    represented_chunks: int


@dataclass(frozen=True)
class AggregateSummary:
    node_id: int
    signature: str
    average_rgba: tuple[int, int, int, int]
    cell_count: int
    chunk_count: int
    image_path: Path
    generated: bool


@dataclass(frozen=True)
class AggregateBuildReport:
    node_count: int
    leaf_count: int
    generated_count: int
    reused_count: int
    total_cells: int
    pending_count: int = 0


class ProductionAggregateHierarchy:
    """Spatial multi-root hierarchy over chunks, grouped within base faces."""

    def __init__(
        self,
        visibility: ProductionVisibilityIndex,
        layout: ProductionChunkLayout,
        leaf_size: int = 4,
    ) -> None:
        if leaf_size < 1:
            raise AggregateLodError("aggregate leaf_size must be positive")
        self.visibility = visibility
        self.layout = layout
        self.leaf_size = int(leaf_size)
        self.parent: dict[int, int] = {}
        nodes: list[AggregateNode | None] = []

        def build_node(chunk_ids: tuple[int, ...]) -> int:
            node_id = len(nodes)
            nodes.append(None)
            center, radius = _cap_from_bounds(visibility.chunks, chunk_ids)
            if len(chunk_ids) <= self.leaf_size:
                nodes[node_id] = AggregateNode(
                    node_id, center, radius, -1, -1, tuple(sorted(chunk_ids))
                )
                return node_id
            axis = _widest_axis(visibility.chunks, chunk_ids)
            ordered = tuple(
                sorted(chunk_ids, key=lambda item: (visibility.chunks[item].center[axis], item))
            )
            midpoint = len(ordered) // 2
            left = build_node(ordered[:midpoint])
            right = build_node(ordered[midpoint:])
            self.parent[left] = node_id
            self.parent[right] = node_id
            nodes[node_id] = AggregateNode(node_id, center, radius, left, right, ())
            return node_id

        roots: list[int] = []
        for face_id in range(20):
            chunk_ids = tuple(
                record.chunk_id for record in layout.records if record.base_face == face_id
            )
            if chunk_ids:
                roots.append(build_node(chunk_ids))
        self.nodes = tuple(node for node in nodes if node is not None)
        self.root_ids = tuple(roots)
        self._descendant_chunks: dict[int, tuple[int, ...]] = {}
        self._leaf_by_chunk: dict[int, int] = {}
        for node in self.nodes:
            if node.is_leaf:
                for chunk_id in node.chunk_ids:
                    self._leaf_by_chunk[chunk_id] = node.node_id

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def descendant_chunks(self, node_id: int) -> tuple[int, ...]:
        cached = self._descendant_chunks.get(node_id)
        if cached is not None:
            return cached
        node = self.nodes[node_id]
        if node.is_leaf:
            result = node.chunk_ids
        else:
            result = tuple(
                sorted(self.descendant_chunks(node.left) + self.descendant_chunks(node.right))
            )
        self._descendant_chunks[node_id] = result
        return result

    def ancestors_for_chunks(self, chunk_ids: Iterable[int]) -> tuple[int, ...]:
        result: set[int] = set()
        for chunk_id in chunk_ids:
            node_id = self._leaf_by_chunk.get(int(chunk_id))
            while node_id is not None:
                result.add(node_id)
                node_id = self.parent.get(node_id)
        return tuple(sorted(result))

    def select(
        self,
        yaw: float,
        pitch: float,
        zoom: float,
        width: int,
        height: int,
        *,
        target_pixels: float = 42.0,
        margin: float = 24.0,
    ) -> AggregateSelection:
        width = max(1, int(width))
        height = max(1, int(height))
        base_radius = max(20.0, min(width, height) * 0.43)
        sphere_radius = base_radius * max(0.25, float(zoom))
        center_x = width / 2.0
        center_y = height / 2.0
        selected: list[int] = []
        visited = 0
        rejected = 0
        stack = list(reversed(self.root_ids))
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            visited += 1
            if not _cap_intersects_viewport(
                node.center, node.angular_radius, yaw, pitch, sphere_radius,
                center_x, center_y, width, height, margin,
            ):
                rejected += 1
                continue
            projected_diameter = (
                2.0 * sphere_radius
                if node.angular_radius >= math.pi / 2.0
                else 2.0 * math.sin(node.angular_radius) * sphere_radius
            )
            if node.is_leaf or projected_diameter <= target_pixels:
                selected.append(node_id)
                continue
            stack.append(node.right)
            stack.append(node.left)
        selected.sort()
        return AggregateSelection(
            node_ids=tuple(selected),
            visited_nodes=visited,
            rejected_nodes=rejected,
            represented_chunks=sum(len(self.descendant_chunks(node_id)) for node_id in selected),
        )


class AggregateLodCache:
    """Regenerable multi-level aggregate color/texture cache."""

    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self.color_cache = DistantLodCache(self.brush_root)
        # A far view resolves several hundred node manifests, and it does so twice
        # per frame: once to decide what still needs building and again to build
        # the proxy batch. Memoising on the manifest's mtime and size turns each
        # of those from a file read plus JSON parse into a stat, and stays correct
        # because invalidation deletes the manifest outright.
        self._summary_cache: dict[tuple[str, int], tuple[tuple[int, int], AggregateSummary]] = {}

    @staticmethod
    def root(map_dir: Path) -> Path:
        return Path(map_dir) / "lod" / "aggregate_v1"

    def manifest_path(self, map_dir: Path, node_id: int) -> Path:
        return self.root(map_dir) / "nodes" / f"node_{node_id:05d}.json"

    def image_path(self, map_dir: Path, node_id: int) -> Path:
        return self.root(map_dir) / "nodes" / f"node_{node_id:05d}.png"

    def existing(self, map_dir: Path, node_id: int) -> AggregateSummary | None:
        manifest = self.manifest_path(map_dir, node_id)
        image = self.image_path(map_dir, node_id)
        key = (str(map_dir), int(node_id))
        try:
            status = manifest.stat()
        except OSError:
            self._summary_cache.pop(key, None)
            return None
        if not image.is_file():
            self._summary_cache.pop(key, None)
            return None
        stamp = (status.st_mtime_ns, status.st_size)
        cached = self._summary_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("version") != AGGREGATE_CACHE_VERSION:
                return None
            color = tuple(int(value) for value in payload["averageRgba"])
            if len(color) != 4:
                return None
            summary = AggregateSummary(
                node_id=int(payload["nodeId"]),
                signature=str(payload["signature"]),
                average_rgba=color,  # type: ignore[arg-type]
                cell_count=int(payload["cellCount"]),
                chunk_count=int(payload["chunkCount"]),
                image_path=image,
                generated=False,
            )
        except Exception:
            self._summary_cache.pop(key, None)
            return None
        self._summary_cache[key] = (stamp, summary)
        return summary

    def build_all(
        self,
        hierarchy: ProductionAggregateHierarchy,
        layout: ProductionChunkLayout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> AggregateBuildReport:
        summaries: dict[int, AggregateSummary] = {}
        generated = 0
        reused = 0
        leaf_count = 0
        for node in reversed(hierarchy.nodes):
            if node.is_leaf:
                leaf_count += 1
                summary = self._build_leaf(
                    node.node_id,
                    node.chunk_ids,
                    layout,
                    session,
                    store,
                    records_by_uid,
                )
            else:
                summary = self._build_parent(
                    node.node_id,
                    summaries[node.left],
                    summaries[node.right],
                    session.map_dir,
                )
            summaries[node.node_id] = summary
            generated += int(summary.generated)
            reused += int(not summary.generated)
        root_summaries = [summaries[node_id] for node_id in hierarchy.root_ids]
        total_cells = sum(item.cell_count for item in root_summaries)
        root_signature = hashlib.sha256(
            b"".join(bytes.fromhex(item.signature) for item in root_summaries)
        ).hexdigest()
        index_payload = {
            "version": AGGREGATE_CACHE_VERSION,
            "format": "HEX_PLANET_AGGREGATE_LOD",
            "layoutHash": layout.stable_hash,
            "visibilityHash": hierarchy.visibility.stable_hash,
            "rootNodes": list(hierarchy.root_ids),
            "rootSignature": root_signature,
            "nodeCount": len(summaries),
            "leafCount": leaf_count,
        }
        _atomic_write_json(self.root(session.map_dir) / "aggregate.json", index_payload)
        return AggregateBuildReport(
            node_count=len(summaries),
            leaf_count=leaf_count,
            generated_count=generated,
            reused_count=reused,
            total_cells=total_cells,
        )

    def ensure_nodes(
        self,
        hierarchy: ProductionAggregateHierarchy,
        layout: ProductionChunkLayout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
        node_ids: Iterable[int],
        *,
        generate_budget: int | None = None,
    ) -> AggregateBuildReport:
        """Build only the currently selected aggregate nodes.

        Each selected node is summarized directly from its descendant Pack chunks.
        This keeps a full f1004 far view practical: the first view can create a few
        coarse nodes without first writing every intermediate node in the tree.
        Saved map edits invalidate the affected ancestor chain, so unchanged nodes
        remain reusable.

        ``generate_budget`` caps how many missing nodes are built in this call.
        A whole-planet view selects several hundred nodes, and summarizing one
        node means reading every Pack chunk beneath it, so building all of them
        inline is a multi-second stall on whatever thread asked for the view.
        Nodes left unbuilt are reported as ``pending_count``; the proxy batch
        already renders a node with no summary as empty, so a budgeted caller
        shows a partial far view now and fills it in over the following frames.
        ``None`` keeps the original build-everything behaviour for the explicit
        cache rebuild and the command line tool.
        """
        generated = 0
        reused = 0
        pending = 0
        total_cells = 0
        remaining = None if generate_budget is None else max(0, int(generate_budget))
        selected = tuple(sorted(set(int(node_id) for node_id in node_ids)))
        for node_id in selected:
            if node_id < 0 or node_id >= hierarchy.node_count:
                raise AggregateLodError(f"aggregate node is outside hierarchy: {node_id}")
            existing = self.existing(session.map_dir, node_id)
            if existing is not None:
                total_cells += existing.cell_count
                reused += 1
                continue
            if remaining is not None and remaining <= 0:
                pending += 1
                continue
            summary = self._build_direct(
                node_id,
                hierarchy.descendant_chunks(node_id),
                layout,
                session,
                store,
                records_by_uid,
            )
            generated += 1
            if remaining is not None:
                remaining -= 1
            total_cells += summary.cell_count
        return AggregateBuildReport(
            node_count=len(selected),
            leaf_count=sum(hierarchy.nodes[node_id].is_leaf for node_id in selected),
            generated_count=generated,
            reused_count=reused,
            total_cells=total_cells,
            pending_count=pending,
        )

    def _build_direct(
        self,
        node_id: int,
        chunk_ids: tuple[int, ...],
        layout: ProductionChunkLayout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> AggregateSummary:
        digest = hashlib.sha256()
        digest.update(b"aggregate-direct-v1")
        digest.update(node_id.to_bytes(4, "little"))
        total_r = total_g = total_b = total_a = 0
        total_cells = 0
        values_by_chunk = store.read_chunk_values_many(session, chunk_ids)
        # Cell values accumulate into one buffer and are hashed in runs instead of
        # two bytes at a time. The digest consumes exactly the same byte stream,
        # so signatures stay compatible with caches built before this change: the
        # buffer is flushed before any brush identity is mixed in, preserving the
        # original interleaving.
        pending = bytearray()
        for chunk_id in chunk_ids:
            values = values_by_chunk[chunk_id]
            if pending:
                digest.update(pending)
                pending.clear()
            digest.update(chunk_id.to_bytes(4, "little"))
            for value in values:
                pending += int(value).to_bytes(2, "little")
                local_id = value & 0x0FFF
                if local_id == 0:
                    color = (63, 73, 82)
                else:
                    entry = session.brush_entries.get(local_id)
                    record = None if entry is None else records_by_uid.get(entry.brush_uid)
                    if record is not None:
                        digest.update(pending)
                        pending.clear()
                        digest.update(record.uid.encode("utf-8"))
                        digest.update(record.content_hash.encode("ascii"))
                    color = self.color_cache.representative_color(record)
                total_r += color[0]
                total_g += color[1]
                total_b += color[2]
                total_a += 255
                total_cells += 1
        if pending:
            digest.update(pending)
            pending.clear()
        return self._write_or_reuse(
            session.map_dir,
            node_id,
            digest.hexdigest(),
            _average(total_r, total_g, total_b, total_a, total_cells),
            total_cells,
            len(chunk_ids),
        )

    def invalidate_chunks(
        self,
        hierarchy: ProductionAggregateHierarchy,
        map_dir: Path,
        chunk_ids: Iterable[int],
    ) -> tuple[int, ...]:
        node_ids = hierarchy.ancestors_for_chunks(chunk_ids)
        for node_id in node_ids:
            self.manifest_path(map_dir, node_id).unlink(missing_ok=True)
            self.image_path(map_dir, node_id).unlink(missing_ok=True)
        (self.root(map_dir) / "aggregate.json").unlink(missing_ok=True)
        return node_ids

    def _build_leaf(
        self,
        node_id: int,
        chunk_ids: tuple[int, ...],
        layout: ProductionChunkLayout,
        session: SphereMapSession,
        store: SphereMapStore,
        records_by_uid: Mapping[str, BrushRecord],
    ) -> AggregateSummary:
        digest = hashlib.sha256()
        digest.update(b"aggregate-leaf-v1")
        digest.update(node_id.to_bytes(4, "little"))
        total_r = total_g = total_b = total_a = 0
        total_cells = 0
        values_by_chunk = store.read_chunk_values_many(session, chunk_ids)
        # Cell values accumulate into one buffer and are hashed in runs instead of
        # two bytes at a time. The digest consumes exactly the same byte stream,
        # so signatures stay compatible with caches built before this change: the
        # buffer is flushed before any brush identity is mixed in, preserving the
        # original interleaving.
        pending = bytearray()
        for chunk_id in chunk_ids:
            values = values_by_chunk[chunk_id]
            if pending:
                digest.update(pending)
                pending.clear()
            digest.update(chunk_id.to_bytes(4, "little"))
            for value in values:
                pending += int(value).to_bytes(2, "little")
                local_id = value & 0x0FFF
                if local_id == 0:
                    color = (63, 73, 82)
                else:
                    entry = session.brush_entries.get(local_id)
                    record = None if entry is None else records_by_uid.get(entry.brush_uid)
                    if record is not None:
                        digest.update(pending)
                        pending.clear()
                        digest.update(record.uid.encode("utf-8"))
                        digest.update(record.content_hash.encode("ascii"))
                    color = self.color_cache.representative_color(record)
                total_r += color[0]
                total_g += color[1]
                total_b += color[2]
                total_a += 255
                total_cells += 1
        if pending:
            digest.update(pending)
            pending.clear()
        signature = digest.hexdigest()
        color_rgba = _average(total_r, total_g, total_b, total_a, total_cells)
        return self._write_or_reuse(
            session.map_dir,
            node_id,
            signature,
            color_rgba,
            total_cells,
            len(chunk_ids),
        )

    def _build_parent(
        self,
        node_id: int,
        left: AggregateSummary,
        right: AggregateSummary,
        map_dir: Path,
    ) -> AggregateSummary:
        digest = hashlib.sha256()
        digest.update(b"aggregate-parent-v1")
        digest.update(node_id.to_bytes(4, "little"))
        digest.update(bytes.fromhex(left.signature))
        digest.update(bytes.fromhex(right.signature))
        total_cells = left.cell_count + right.cell_count
        color = tuple(
            round(
                (left.average_rgba[index] * left.cell_count + right.average_rgba[index] * right.cell_count)
                / max(1, total_cells)
            )
            for index in range(4)
        )
        return self._write_or_reuse(
            map_dir,
            node_id,
            digest.hexdigest(),
            color,  # type: ignore[arg-type]
            total_cells,
            left.chunk_count + right.chunk_count,
        )

    def _write_or_reuse(
        self,
        map_dir: Path,
        node_id: int,
        signature: str,
        color: tuple[int, int, int, int],
        cell_count: int,
        chunk_count: int,
    ) -> AggregateSummary:
        existing = self.existing(map_dir, node_id)
        if existing is not None and existing.signature == signature:
            return existing
        image_path = self.image_path(map_dir, node_id)
        image = PixelImage(
            AGGREGATE_SIDE,
            AGGREGATE_SIDE,
            4,
            bytes(color) * (AGGREGATE_SIDE * AGGREGATE_SIDE),
        )
        # The aggregate cache is derived from the Pack map and is rebuilt on
        # demand, so neither of these two writes needs an fsync. On Windows that
        # syscall measured 43 ms per call, which was 86 ms of the 345 ms it took
        # to build one node - more than the Pack reads it summarises.
        atomic_write_png(image_path, image, durable=False)
        payload = {
            "version": AGGREGATE_CACHE_VERSION,
            "nodeId": node_id,
            "signature": signature,
            "averageRgba": list(color),
            "cellCount": cell_count,
            "chunkCount": chunk_count,
            "image": image_path.name,
        }
        _atomic_write_json(self.manifest_path(map_dir, node_id), payload)
        return AggregateSummary(
            node_id=node_id,
            signature=signature,
            average_rgba=color,
            cell_count=cell_count,
            chunk_count=chunk_count,
            image_path=image_path,
            generated=True,
        )


def build_aggregate_proxy_batch(
    hierarchy: ProductionAggregateHierarchy,
    selection: AggregateSelection,
    topology_hash: str,
    layout_hash: str,
    map_dir: Path,
    cache: AggregateLodCache,
) -> GpuRenderBatch:
    layers: list[GpuTextureLayer] = [
        GpuTextureLayer(0, EMPTY_LAYER_KEY, None, "", None, "empty"),
        GpuTextureLayer(1, MISSING_LAYER_KEY, None, "", None, "missing"),
    ]
    layer_by_signature: dict[str, int] = {}
    instances: list[GpuInstance] = []
    for node_id in selection.node_ids:
        node = hierarchy.nodes[node_id]
        summary = cache.existing(map_dir, node_id)
        if summary is None:
            layer = 0
        else:
            key = f"aggregate:{node_id}:{summary.signature}"
            layer = layer_by_signature.get(summary.signature, -1)
            if layer < 0:
                layer = len(layers)
                layer_by_signature[summary.signature] = layer
                layers.append(
                    GpuTextureLayer(
                        layer,
                        key,
                        None,
                        summary.signature,
                        summary.image_path,
                        "brush",
                    )
                )
        instances.append(
            GpuInstance(
                cell_id=AGGREGATE_CELL_BASE + node_id,
                corners=_proxy_corners(node.center, node.angular_radius),
                texture_layer=layer,
                rotation=0,
            )
        )
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(topology_hash))
    digest.update(bytes.fromhex(layout_hash))
    digest.update(b"aggregate-proxy-v1")
    for instance in instances:
        digest.update(instance.packed())
        digest.update(instance.cell_id.to_bytes(4, "little"))
    for layer in layers:
        digest.update(layer.key.encode("utf-8"))
    return GpuRenderBatch(
        version=GPU_BATCH_VERSION,
        lod_level=3,
        effective_size=AGGREGATE_EFFECTIVE,
        padded_size=AGGREGATE_SIDE,
        topology_hash=topology_hash,
        layout_hash=layout_hash,
        instances=tuple(instances),
        texture_layers=tuple(layers),
        stable_hash=digest.hexdigest(),
    )


def _proxy_corners(
    center: tuple[float, float, float], angular_radius: float
) -> tuple[tuple[float, float, float], ...]:
    tangent_u, tangent_v = _basis(center)
    distance = math.tan(max(0.001, min(1.2, angular_radius)))
    corners = []
    for index in range(6):
        angle = math.radians(index * 60.0)
        local_u = math.cos(angle) * distance
        local_v = math.sin(angle) * distance
        corners.append(
            _normalize(
                (
                    center[0] + tangent_u[0] * local_u + tangent_v[0] * local_v,
                    center[1] + tangent_u[1] * local_u + tangent_v[1] * local_v,
                    center[2] + tangent_u[2] * local_u + tangent_v[2] * local_v,
                )
            )
        )
    return tuple(corners)


def _basis(center: tuple[float, float, float]):
    reference = (0.0, 1.0, 0.0) if abs(center[1]) < 0.9 else (1.0, 0.0, 0.0)
    tangent_u = _normalize(_cross(reference, center))
    tangent_v = _normalize(_cross(center, tangent_u))
    return tangent_u, tangent_v


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(value):
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1e-15:
        raise AggregateLodError("Cannot normalize aggregate basis")
    return tuple(component / length for component in value)


def _average(r: int, g: int, b: int, a: int, count: int) -> tuple[int, int, int, int]:
    if count <= 0:
        return (63, 73, 82, 255)
    return (
        round(r / count),
        round(g / count),
        round(b / count),
        round(a / count),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
