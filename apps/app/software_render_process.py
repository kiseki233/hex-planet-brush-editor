from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

from .png_pixels import PixelImage
from .software_globe import render_textured_globe_view_ppm


class SoftwareRenderProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class SoftwareRenderRequest:
    generation: int
    key: tuple[object, ...]
    yaw: float
    pitch: float
    width: int
    height: int
    center_x: float
    center_y: float
    radius: float
    block_size: int
    preview_scale: int


@dataclass(frozen=True)
class SoftwareRenderResult:
    generation: int
    key: tuple[object, ...]
    ppm: bytes
    preview_scale: int
    error: str | None = None


def _replace_queue_item(target: Any, item: object) -> None:
    for _attempt in range(8):
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
        try:
            target.put(item, block=True, timeout=0.05)
            return
        except queue.Full:
            time.sleep(0.002)
    raise SoftwareRenderProcessError("Unable to replace the bounded render queue item")


def _render_worker(
    texture: PixelImage,
    request_queue: Any,
    result_queue: Any,
) -> None:
    while True:
        request = request_queue.get()
        if request is None:
            return
        latest = request
        while True:
            try:
                candidate = request_queue.get_nowait()
            except queue.Empty:
                break
            if candidate is None:
                return
            latest = candidate
        try:
            ppm = render_textured_globe_view_ppm(
                texture,
                latest.yaw,
                latest.pitch,
                latest.width,
                latest.height,
                latest.center_x,
                latest.center_y,
                latest.radius,
                block_size=latest.block_size,
            )
            result = SoftwareRenderResult(
                generation=latest.generation,
                key=latest.key,
                ppm=ppm,
                preview_scale=latest.preview_scale,
            )
        except Exception as exc:
            result = SoftwareRenderResult(
                generation=latest.generation,
                key=latest.key,
                ppm=b"",
                preview_scale=latest.preview_scale,
                error=str(exc),
            )
        _replace_queue_item(result_queue, result)


class SoftwareGlobeRenderProcess:
    """Persistent process that isolates software globe rendering from Tk's GIL."""

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._request_queue: Any | None = None
        self._result_queue: Any | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def set_texture(self, texture: PixelImage) -> None:
        self.close()
        request_queue = self._context.Queue(maxsize=1)
        result_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_render_worker,
            args=(texture, request_queue, result_queue),
            name="hexplanet-software-render-process",
            daemon=True,
        )
        process.start()
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._process = process

    def submit(self, request: SoftwareRenderRequest) -> None:
        if not self.running or self._request_queue is None:
            raise SoftwareRenderProcessError("Software render process is not running")
        _replace_queue_item(self._request_queue, request)

    def poll_latest(self) -> SoftwareRenderResult | None:
        if self._result_queue is None:
            return None
        latest = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except queue.Empty:
                return latest
            except (EOFError, OSError):
                return latest

    def close(self) -> None:
        process = self._process
        request_queue = self._request_queue
        result_queue = self._result_queue
        self._process = None
        self._request_queue = None
        self._result_queue = None
        if process is not None:
            if process.is_alive() and request_queue is not None:
                try:
                    _replace_queue_item(request_queue, None)
                except Exception:
                    pass
                process.join(timeout=0.75)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.75)
        for item in (request_queue, result_queue):
            if item is None:
                continue
            try:
                item.cancel_join_thread()
            except Exception:
                pass
            try:
                item.close()
            except Exception:
                pass
