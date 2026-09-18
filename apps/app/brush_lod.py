from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .brush_catalog import BrushRecord
from .png_pixels import add_edge_padding, atomic_write_png, read_png_pixels, resize_nearest

LOD_EFFECTIVE_SIZES = (512, 256, 128, 64)
LOD_PADDING = 4
LOD_CACHE_VERSION = 1


class BrushLodError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrushLodInfo:
    uid: str
    content_hash: str
    level: int
    effective_size: int
    padded_size: int
    path: Path
    generated: bool


class BrushLodCache:
    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self.cache_root = self.brush_root / ".catalog" / "cache" / "lod_v1"

    @staticmethod
    def effective_size(level: int) -> int:
        try:
            return LOD_EFFECTIVE_SIZES[level]
        except IndexError as exc:
            raise BrushLodError(f"Unsupported brush LOD level: {level}") from exc

    @staticmethod
    def padded_size(level: int) -> int:
        return BrushLodCache.effective_size(level) + LOD_PADDING * 2

    def path_for(self, record: BrushRecord, level: int) -> Path:
        effective = self.effective_size(level)
        safe_uid = record.uid.replace("/", "_").replace("\\", "_")
        return self.cache_root / safe_uid / record.content_hash / f"lod{level}_{effective + 8}.png"

    def existing_path(self, record: BrushRecord, level: int) -> Path | None:
        path = self.path_for(record, level)
        return path if path.is_file() else None

    def ensure(self, record: BrushRecord, level: int) -> BrushLodInfo:
        if record.state != "active":
            raise BrushLodError(f"Cannot generate LOD for missing brush: {record.uid}")
        target = self.path_for(record, level)
        if target.is_file():
            return BrushLodInfo(
                record.uid,
                record.content_hash,
                level,
                self.effective_size(level),
                self.padded_size(level),
                target,
                False,
            )
        source_path = self.brush_root / record.relative_path
        try:
            source = read_png_pixels(source_path)
        except Exception as exc:
            raise BrushLodError(f"Cannot decode brush {record.relative_path}: {exc}") from exc
        if source.width != 512 or source.height != 512:
            raise BrushLodError(
                f"Brush size changed after catalog scan: {record.relative_path} is {source.width}x{source.height}"
            )
        effective = self.effective_size(level)
        resized = resize_nearest(source, effective, effective)
        padded = add_edge_padding(resized, LOD_PADDING)
        atomic_write_png(target, padded)
        self._write_manifest(record)
        self.prune_superseded(record)
        return BrushLodInfo(
            record.uid,
            record.content_hash,
            level,
            effective,
            effective + 8,
            target,
            True,
        )

    def prune_superseded(self, record: BrushRecord) -> int:
        """Delete cached levels this brush generated under earlier content.

        The cache path embeds the content hash, so replacing a brush image in
        place strands its whole previous LOD set - four PNGs per brush, which
        across a full library replacement is the entire old cache. Nothing else
        refers to those directories: layer keys and manifests are always derived
        from the record's current hash.
        """
        safe_uid = record.uid.replace("/", "_").replace("\\", "_")
        brush_dir = self.cache_root / safe_uid
        if not brush_dir.is_dir():
            return 0
        removed = 0
        for child in brush_dir.iterdir():
            if not child.is_dir() or child.name == record.content_hash:
                continue
            try:
                shutil.rmtree(child)
            except OSError:
                continue
            removed += 1
        return removed

    def ensure_all(self, record: BrushRecord) -> tuple[BrushLodInfo, ...]:
        return tuple(self.ensure(record, level) for level in range(len(LOD_EFFECTIVE_SIZES)))

    def _write_manifest(self, record: BrushRecord) -> None:
        directory = self.cache_root / record.uid / record.content_hash
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LOD_CACHE_VERSION,
            "brushUid": record.uid,
            "contentHash": record.content_hash,
            "sourcePath": record.relative_path,
            "padding": LOD_PADDING,
            "levels": [
                {
                    "level": level,
                    "effectiveSize": size,
                    "paddedSize": size + LOD_PADDING * 2,
                    "file": f"lod{level}_{size + LOD_PADDING * 2}.png",
                }
                for level, size in enumerate(LOD_EFFECTIVE_SIZES)
            ],
        }
        target = directory / "manifest.json"
        handle, temporary_name = tempfile.mkstemp(prefix="manifest_", suffix=".json.tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


class BrushLodPolicy:
    """Four texture LOD levels plus level 4 for distant flat-color rendering."""

    def __init__(self, initial_level: int = 0) -> None:
        if initial_level < 0 or initial_level > 4:
            raise ValueError("initial_level must be between 0 and 4")
        self.level = initial_level

    def update(self, visible_cell_count: int) -> int:
        count = max(0, visible_cell_count)
        while self.level < 4 and count > self._out_threshold(self.level):
            self.level += 1
        while self.level > 0 and count < self._in_threshold(self.level):
            self.level -= 1
        return self.level

    @staticmethod
    def _out_threshold(level: int) -> int:
        return (17, 33, 65, 129)[level] ** 2

    @staticmethod
    def _in_threshold(level: int) -> int:
        return (15, 31, 63, 127)[level - 1] ** 2

    @staticmethod
    def description(level: int) -> str:
        if 0 <= level < 4:
            return f"LOD{level} {LOD_EFFECTIVE_SIZES[level]}×{LOD_EFFECTIVE_SIZES[level]}"
        return "远景已保存地表（逐格纹理暂停，仍可绘制）"


@dataclass(frozen=True)
class BrushLodBuildResult:
    uid: str
    content_hash: str
    level: int
    path: Path | None
    generated: bool
    error: str | None


class AsyncBrushLodBuilder:
    def __init__(self, cache: BrushLodCache, max_workers: int = 2, max_inflight: int = 8) -> None:
        import concurrent.futures

        if max_workers < 1 or max_inflight < 1:
            raise ValueError("max_workers and max_inflight must be positive")
        self.cache = cache
        self.max_inflight = max_inflight
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hexplanet-lod",
        )
        self.pending: dict[tuple[str, str, int], object] = {}
        self.failed: dict[tuple[str, str, int], str] = {}
        self.closed = False

    def request(self, record: BrushRecord, level: int) -> Path | None:
        existing = self.cache.existing_path(record, level)
        if existing is not None:
            return existing
        if self.closed:
            return None
        key = (record.uid, record.content_hash, level)
        if key in self.pending or key in self.failed:
            return None
        if len(self.pending) >= self.max_inflight:
            return None
        self.pending[key] = self.executor.submit(self.cache.ensure, record, level)
        return None

    def poll(self) -> tuple[BrushLodBuildResult, ...]:
        results: list[BrushLodBuildResult] = []
        for key, future in tuple(self.pending.items()):
            if not future.done():
                continue
            self.pending.pop(key, None)
            uid, content_hash, level = key
            try:
                info = future.result()
            except Exception as exc:
                message = str(exc)
                self.failed[key] = message
                results.append(
                    BrushLodBuildResult(uid, content_hash, level, None, False, message)
                )
            else:
                results.append(
                    BrushLodBuildResult(
                        uid,
                        content_hash,
                        level,
                        info.path,
                        info.generated,
                        None,
                    )
                )
        return tuple(results)

    def pending_count(self) -> int:
        return len(self.pending)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()
        self.failed.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)
