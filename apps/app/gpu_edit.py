from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from typing import Iterable

from .gpu_batch import GpuRenderBatch
from .gpu_stream import GpuInstancePatch
from .topology import DualTopology
from .i18n import t


class GpuEditError(ValueError):
    pass


@dataclass(frozen=True)
class GpuToolState:
    tool: str = "paint"
    brush_uid: str | None = None
    last_known_path: str = ""
    rotation: int = 0
    brush_group: str = ""
    brush_diameter: int = 1

    def validated(self) -> "GpuToolState":
        if self.tool not in {"paint", "erase"}:
            raise GpuEditError(f"Unsupported GPU edit tool: {self.tool}")
        if self.rotation < 0 or self.rotation > 5:
            raise GpuEditError("GPU edit rotation must be between 0 and 5")
        if self.brush_diameter < 1 or self.brush_diameter > 500:
            raise GpuEditError("GPU brush diameter must be between 1 and 500 cells")
        return self


@dataclass(frozen=True)
class GpuEditRequest:
    request_id: int
    cell_id: int
    state: GpuToolState
    stroke_id: int = 0
    phase: str = "point"


@dataclass(frozen=True)
class GpuSaveRequest:
    request_id: int


@dataclass(frozen=True)
class GpuResyncRequest:
    request_id: int
    reason: str


@dataclass(frozen=True)
class GpuViewRequest:
    request_id: int
    yaw: float
    pitch: float
    zoom: float
    width: int
    height: int
    interactive: bool = False


GpuRequest = GpuEditRequest | GpuSaveRequest | GpuResyncRequest | GpuViewRequest


@dataclass(frozen=True)
class GpuCellPatch:
    request_id: int
    cell_id: int
    texture_key: str
    rotation: int
    message: str
    pixels_rgba: bytes | None = None
    texture_width: int = 0
    texture_height: int = 0


@dataclass(frozen=True)
class GpuStatusPatch:
    request_id: int
    success: bool
    message: str


@dataclass(frozen=True)
class GpuSurfaceTexturePatch:
    request_id: int
    width: int
    height: int
    channels: int
    pixels: bytes
    message: str = t("星球缩略地表已更新")


@dataclass(frozen=True)
class GpuSurfaceRegionPatch:
    request_id: int
    x: int
    y: int
    width: int
    height: int
    channels: int
    pixels: bytes
    message: str = t("L5 远景局部纹理已更新")


@dataclass(frozen=True)
class GpuTextureUpload:
    layer: int
    key: str
    width: int
    height: int
    pixels_rgba: bytes
    replace_existing: bool = False
    source_path: str | None = None


@dataclass(frozen=True)
class GpuStreamPatch:
    request_id: int
    active_chunk_ids: tuple[int, ...]
    instance_count: int
    removed_cell_ids: tuple[int, ...]
    added: tuple[GpuInstancePatch, ...]
    changed: tuple[GpuInstancePatch, ...]
    texture_uploads: tuple[GpuTextureUpload, ...]
    message: str
    lod_level: int = 0
    padded_size: int = 0
    reset_textures: bool = False
    released_texture_layers: tuple[int, ...] = ()
    has_more: bool = False
    editable: bool = True
    loaded_chunk_count: int = 0
    total_chunk_count: int = 0
    remaining_chunk_count: int = 0


@dataclass(frozen=True)
class GpuBatchResetPatch:
    request_id: int
    batch: GpuRenderBatch
    message: str
    editable: bool = True
    has_more: bool = False
    loaded_chunk_count: int = 0
    total_chunk_count: int = 0
    remaining_chunk_count: int = 0
    texture_uploads: tuple[GpuTextureUpload, ...] = ()


GpuPatch = (
    GpuCellPatch
    | GpuStatusPatch
    | GpuSurfaceTexturePatch
    | GpuSurfaceRegionPatch
    | GpuStreamPatch
    | GpuBatchResetPatch
)


class GpuEditBridge:
    """Thread-safe command bridge between the native window and Tk owner.

    The native thread never mutates the map session or Pack files. It submits
    immutable requests. The Tk owner applies them and returns render patches.
    """

    def __init__(
        self,
        initial_state: GpuToolState | None = None,
        known_texture_keys: Iterable[str] = (),
    ) -> None:
        self._state_lock = threading.Lock()
        self._state = (initial_state or GpuToolState()).validated()
        self._requests: queue.Queue[GpuRequest] = queue.Queue()
        self._patches: queue.Queue[GpuPatch] = queue.Queue()
        self._closed = threading.Event()
        self._sequence_lock = threading.Lock()
        self._next_request_id = 1
        self._texture_lock = threading.Lock()
        self._known_texture_keys = set(known_texture_keys)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        self._closed.set()

    def set_tool_state(self, state: GpuToolState) -> None:
        state = state.validated()
        with self._state_lock:
            self._state = state

    def tool_state(self) -> GpuToolState:
        with self._state_lock:
            return self._state

    def cycle_rotation(self) -> GpuToolState:
        with self._state_lock:
            self._state = GpuToolState(
                tool=self._state.tool,
                brush_uid=self._state.brush_uid,
                last_known_path=self._state.last_known_path,
                rotation=(self._state.rotation + 1) % 6,
                brush_group=self._state.brush_group,
                brush_diameter=self._state.brush_diameter,
            )
            return self._state

    def set_tool(self, tool: str) -> GpuToolState:
        with self._state_lock:
            self._state = GpuToolState(
                tool=tool,
                brush_uid=self._state.brush_uid,
                last_known_path=self._state.last_known_path,
                rotation=self._state.rotation,
                brush_group=self._state.brush_group,
                brush_diameter=self._state.brush_diameter,
            ).validated()
            return self._state

    def submit_edit(
        self,
        cell_id: int,
        *,
        stroke_id: int = 0,
        phase: str = "point",
    ) -> int:
        if self.closed:
            raise GpuEditError("GPU edit bridge is closed")
        if phase not in {"point", "start", "move", "end"}:
            raise GpuEditError(f"Unsupported GPU stroke phase: {phase}")
        request_id = self._allocate_request_id()
        self._requests.put(
            GpuEditRequest(
                request_id, int(cell_id), self.tool_state(), int(stroke_id), phase
            )
        )
        return request_id

    def submit_stroke_end(self, stroke_id: int) -> int:
        return self.submit_edit(-1, stroke_id=stroke_id, phase="end")

    def submit_save(self) -> int:
        if self.closed:
            raise GpuEditError("GPU edit bridge is closed")
        request_id = self._allocate_request_id()
        self._requests.put(GpuSaveRequest(request_id))
        return request_id

    def submit_resync(self, reason: str) -> int:
        if self.closed:
            raise GpuEditError("GPU edit bridge is closed")
        request_id = self._allocate_request_id()
        self._requests.put(GpuResyncRequest(request_id, str(reason)))
        return request_id

    def submit_view(
        self,
        yaw: float,
        pitch: float,
        zoom: float,
        width: int,
        height: int,
        *,
        interactive: bool = False,
    ) -> int:
        if self.closed:
            raise GpuEditError("GPU edit bridge is closed")
        request_id = self._allocate_request_id()
        self._requests.put(
            GpuViewRequest(
                request_id, float(yaw), float(pitch), float(zoom),
                max(1, int(width)), max(1, int(height)), bool(interactive)
            )
        )
        return request_id

    def poll_requests(self, limit: int = 64) -> tuple[GpuRequest, ...]:
        result: list[GpuRequest] = []
        for _ in range(max(0, limit)):
            try:
                result.append(self._requests.get_nowait())
            except queue.Empty:
                break
        return tuple(result)

    def push_patch(self, patch: GpuPatch) -> None:
        if not self.closed:
            self._patches.put(patch)

    def poll_patches(self, limit: int = 64) -> tuple[GpuPatch, ...]:
        result: list[GpuPatch] = []
        for _ in range(max(0, limit)):
            try:
                result.append(self._patches.get_nowait())
            except queue.Empty:
                break
        return tuple(result)

    def claim_texture_key(self, key: str) -> bool:
        """Return True once for each texture key that needs an upload payload."""
        with self._texture_lock:
            if key in self._known_texture_keys:
                return False
            self._known_texture_keys.add(key)
            return True

    def forget_texture_key(self, key: str) -> None:
        with self._texture_lock:
            self._known_texture_keys.discard(key)

    def _allocate_request_id(self) -> int:
        with self._sequence_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id


@dataclass(frozen=True)
class GpuPickResult:
    cell_id: int
    instance_index: int | None
    world_direction: tuple[float, float, float]


class SphericalCellPicker:
    """Fast exact-nearest-center picker over the dual topology graph."""

    def __init__(self, topology: DualTopology, batch: GpuRenderBatch, seed_count: int = 64) -> None:
        if batch.topology_hash != topology.stable_hash:
            raise GpuEditError("GPU batch and topology hashes do not match")
        self.topology = topology
        self.instance_by_cell = {
            instance.cell_id: index for index, instance in enumerate(batch.instances)
        }
        count = topology.cell_count
        if count == 0:
            raise GpuEditError("Topology has no cells")
        target = max(1, min(seed_count, count))
        step = max(1, count // target)
        seeds = list(range(0, count, step))[:target]
        seeds.extend(topology.pentagon_ids)
        self.seeds = tuple(dict.fromkeys(seeds))

    def update_instances(self, instances) -> None:
        self.instance_by_cell = {
            instance.cell_id: index for index, instance in enumerate(instances)
        }

    def pick_screen(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        yaw: float,
        pitch: float,
        zoom: float,
    ) -> GpuPickResult | None:
        direction = screen_to_world_direction(x, y, width, height, yaw, pitch, zoom)
        if direction is None:
            return None
        cell_id = self.nearest_cell(direction)
        return GpuPickResult(cell_id, self.instance_by_cell.get(cell_id), direction)

    def nearest_cell(self, direction: tuple[float, float, float]) -> int:
        direction = _normalize(direction)
        ranked = sorted(
            self.seeds,
            key=lambda cell_id: _dot(self.topology.cell_centers[cell_id], direction),
            reverse=True,
        )[:4]
        best_cell = ranked[0]
        best_dot = -2.0
        for seed in ranked:
            current = seed
            current_dot = _dot(self.topology.cell_centers[current], direction)
            visited: set[int] = set()
            while current not in visited:
                visited.add(current)
                next_cell = current
                next_dot = current_dot
                for neighbor in self.topology.neighbors[current]:
                    value = _dot(self.topology.cell_centers[neighbor], direction)
                    if value > next_dot + 1e-15:
                        next_cell = neighbor
                        next_dot = value
                if next_cell == current:
                    break
                current = next_cell
                current_dot = next_dot
            if current_dot > best_dot:
                best_cell = current
                best_dot = current_dot
        return best_cell


def screen_to_world_direction(
    x: float,
    y: float,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
) -> tuple[float, float, float] | None:
    width = max(1, int(width))
    height = max(1, int(height))
    scale = 0.86 * max(0.8, min(512.0, float(zoom)))
    aspect = width / height
    ndc_x = (float(x) / width) * 2.0 - 1.0
    ndc_y = 1.0 - (float(y) / height) * 2.0
    view_x = ndc_x * aspect / scale
    view_y = ndc_y / scale
    radius_sq = view_x * view_x + view_y * view_y
    if radius_sq > 1.0:
        return None
    view_z = math.sqrt(max(0.0, 1.0 - radius_sq))

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    yawed_x = view_x
    yawed_y = view_y * cp + view_z * sp
    yawed_z = -view_y * sp + view_z * cp

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    world_x = yawed_x * cy - yawed_z * sy
    world_y = yawed_y
    world_z = yawed_x * sy + yawed_z * cy
    return _normalize((world_x, world_y, world_z))


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(value, value))
    if length <= 1e-15:
        raise GpuEditError("Cannot normalize a zero-length direction")
    return value[0] / length, value[1] / length, value[2] / length
