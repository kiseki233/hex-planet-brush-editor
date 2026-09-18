from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .png_info import PngValidationError, validate_brush_png
from .png_pixels import PngPixelError, read_png_pixels, sample_average_rgb

CATALOG_VERSION = 1
MAX_ACTIVE_BRUSHES = 4095


class BrushCatalogError(RuntimeError):
    pass


@dataclass
class BrushRecord:
    uid: str
    relative_path: str
    content_hash: str
    file_size: int
    width: int
    height: int
    color_mode: str
    category_path: str
    modified_time_ns: int
    state: str
    last_seen_utc: int
    average_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class InvalidBrush:
    relative_path: str
    reason: str


@dataclass
class ScanResult:
    records: list[BrushRecord]
    invalid: list[InvalidBrush]
    active_count: int
    missing_count: int
    scanned_png_count: int
    catalog_path: Path

    @property
    def active_records(self) -> list[BrushRecord]:
        return [record for record in self.records if record.state == "active"]


@dataclass(frozen=True)
class _Candidate:
    relative_path: str
    absolute_path: Path
    content_hash: str
    file_size: int
    width: int
    height: int
    color_mode: str
    category_path: str
    modified_time_ns: int


class BrushCatalog:
    def __init__(self, brush_root: str | Path) -> None:
        self.brush_root = Path(brush_root)
        self.catalog_root = self.brush_root / ".catalog"
        self.catalog_path = self.catalog_root / "brushes.json"
        self.cache_root = self.catalog_root / "cache"

    def scan(self) -> ScanResult:
        self.brush_root.mkdir(parents=True, exist_ok=True)
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        existing = self._load_records()
        existing_by_path = {record.relative_path: record for record in existing}
        candidates, invalid, scanned_png_count = self._scan_candidates(existing_by_path)
        now_utc = int(time.time())
        average_by_hash = {
            record.content_hash: record.average_rgb
            for record in existing
            if self._valid_average_rgb(record.average_rgb)
        }

        unmatched_existing = {record.uid: record for record in existing}
        accepted: list[BrushRecord] = []
        unmatched_candidates: list[_Candidate] = []

        for candidate in candidates:
            record = existing_by_path.get(candidate.relative_path)
            if record is None:
                unmatched_candidates.append(candidate)
                continue
            accepted.append(
                self._updated_record(
                    record.uid, candidate, now_utc, average_by_hash
                )
            )
            unmatched_existing.pop(record.uid, None)

        # Files that no longer sit at a known path are matched to retired records
        # by content, which is what lets a brush keep its uid - and therefore the
        # cells already painted with it - across a move or rename.
        #
        # Pairing is done one for one within each content hash. Requiring the
        # hash to be unique on both sides used to be the condition, which meant
        # relocating several byte-identical brushes in a single refresh retired
        # all of them and minted fresh uids: every cell painted with any of them
        # rendered as missing texture. Since the files are identical, which
        # record claims which path cannot change what is drawn, so a stable
        # arbitrary pairing is both safe and strictly better than giving up.
        existing_hash_pools: dict[str, list[BrushRecord]] = {}
        for record in sorted(
            unmatched_existing.values(),
            key=lambda item: (item.relative_path.casefold(), item.uid),
        ):
            existing_hash_pools.setdefault(record.content_hash, []).append(record)

        remaining_candidates: list[_Candidate] = []
        for candidate in sorted(
            unmatched_candidates, key=lambda item: item.relative_path.casefold()
        ):
            pool = existing_hash_pools.get(candidate.content_hash)
            if pool:
                moved_record = pool.pop(0)
                accepted.append(
                    self._updated_record(
                        moved_record.uid, candidate, now_utc, average_by_hash
                    )
                )
                unmatched_existing.pop(moved_record.uid, None)
            else:
                remaining_candidates.append(candidate)

        if len(accepted) > MAX_ACTIVE_BRUSHES:
            raise BrushCatalogError(
                f"Brush catalog already contains more than {MAX_ACTIVE_BRUSHES} active records"
            )

        available_slots = max(0, MAX_ACTIVE_BRUSHES - len(accepted))
        for candidate in remaining_candidates[:available_slots]:
            accepted.append(
                self._updated_record(
                    str(uuid.uuid4()), candidate, now_utc, average_by_hash
                )
            )

        for candidate in remaining_candidates[available_slots:]:
            invalid.append(InvalidBrush(candidate.relative_path, "brush_limit_exceeded"))

        missing_records: list[BrushRecord] = []
        for record in unmatched_existing.values():
            record.state = "missing"
            missing_records.append(record)

        records = sorted(accepted + missing_records, key=lambda item: (item.state != "active", item.relative_path.casefold(), item.uid))
        self._save_records(records)
        return ScanResult(
            records=records,
            invalid=sorted(invalid, key=lambda item: item.relative_path.casefold()),
            active_count=sum(1 for item in records if item.state == "active"),
            missing_count=sum(1 for item in records if item.state == "missing"),
            scanned_png_count=scanned_png_count,
            catalog_path=self.catalog_path,
        )

    def _scan_candidates(
        self,
        existing_by_path: dict[str, BrushRecord],
    ) -> tuple[list[_Candidate], list[InvalidBrush], int]:
        candidates: list[_Candidate] = []
        invalid: list[InvalidBrush] = []
        scanned_png_count = 0

        for current_root, directory_names, file_names in os.walk(self.brush_root):
            directory_names[:] = [
                name for name in directory_names if not name.startswith(".") and name != ".catalog"
            ]
            current_path = Path(current_root)
            for file_name in sorted(file_names, key=str.casefold):
                if not file_name.lower().endswith(".png"):
                    continue
                scanned_png_count += 1
                absolute_path = current_path / file_name
                relative_path = absolute_path.relative_to(self.brush_root).as_posix()
                try:
                    stat = absolute_path.stat()
                except OSError as exc:
                    invalid.append(InvalidBrush(relative_path, f"io_error:{exc}"))
                    continue

                category_path = Path(relative_path).parent.as_posix()
                if category_path == ".":
                    category_path = ""
                previous = existing_by_path.get(relative_path)
                if (
                    previous is not None
                    and previous.file_size == stat.st_size
                    and previous.modified_time_ns == stat.st_mtime_ns
                    and previous.width == 512
                    and previous.height == 512
                    and previous.color_mode in {"RGB", "RGBA"}
                ):
                    content_hash = previous.content_hash
                    file_size = previous.file_size
                    width = previous.width
                    height = previous.height
                    color_mode = previous.color_mode
                else:
                    try:
                        info = validate_brush_png(absolute_path)
                    except PngValidationError as exc:
                        invalid.append(InvalidBrush(relative_path, str(exc)))
                        continue
                    content_hash = info.sha256
                    file_size = info.file_size
                    width = info.width
                    height = info.height
                    color_mode = info.color_mode
                candidates.append(
                    _Candidate(
                        relative_path=relative_path,
                        absolute_path=absolute_path,
                        content_hash=content_hash,
                        file_size=file_size,
                        width=width,
                        height=height,
                        color_mode=color_mode,
                        category_path=category_path,
                        modified_time_ns=stat.st_mtime_ns,
                    )
                )

        candidates.sort(key=lambda item: item.relative_path.casefold())
        return candidates, invalid, scanned_png_count

    def _updated_record(
        self,
        uid: str,
        candidate: _Candidate,
        now_utc: int,
        average_by_hash: dict[str, tuple[int, int, int] | None],
    ) -> BrushRecord:
        average_rgb = average_by_hash.get(candidate.content_hash)
        if not self._valid_average_rgb(average_rgb):
            try:
                average_rgb = sample_average_rgb(
                    read_png_pixels(candidate.absolute_path)
                )
            except PngPixelError:
                # Keep compatibility with catalogs produced before representative
                # colors were persisted. The distant renderer retains its
                # on-demand fallback for an unusual PNG the scanner can validate
                # but the pixel decoder cannot read.
                average_rgb = None
            average_by_hash[candidate.content_hash] = average_rgb
        return BrushRecord(
            uid=uid,
            relative_path=candidate.relative_path,
            content_hash=candidate.content_hash,
            file_size=candidate.file_size,
            width=candidate.width,
            height=candidate.height,
            color_mode=candidate.color_mode,
            category_path=candidate.category_path,
            modified_time_ns=candidate.modified_time_ns,
            state="active",
            last_seen_utc=now_utc,
            average_rgb=average_rgb,
        )

    @staticmethod
    def _valid_average_rgb(value: object) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(isinstance(channel, int) and 0 <= channel <= 255 for channel in value)
        )

    def _load_records(self) -> list[BrushRecord]:
        if not self.catalog_path.exists():
            return []
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BrushCatalogError(f"Cannot read brush catalog: {exc}") from exc

        if payload.get("version") != CATALOG_VERSION:
            raise BrushCatalogError(
                f"Unsupported brush catalog version: {payload.get('version')}"
            )

        try:
            records: list[BrushRecord] = []
            for item in payload.get("records", []):
                record_payload = dict(item)
                average_rgb = record_payload.get("average_rgb")
                if self._valid_average_rgb(average_rgb):
                    record_payload["average_rgb"] = tuple(average_rgb)
                else:
                    record_payload["average_rgb"] = None
                records.append(BrushRecord(**record_payload))
        except (TypeError, KeyError, ValueError) as exc:
            raise BrushCatalogError(f"Invalid brush catalog record: {exc}") from exc

        seen_uids: set[str] = set()
        seen_paths: set[str] = set()
        for record in records:
            if record.uid in seen_uids:
                raise BrushCatalogError(f"Duplicate brush UID in catalog: {record.uid}")
            seen_uids.add(record.uid)
            if record.state == "active":
                if record.relative_path in seen_paths:
                    raise BrushCatalogError(
                        f"Duplicate active brush path in catalog: {record.relative_path}"
                    )
                seen_paths.add(record.relative_path)
        return records

    def _save_records(self, records: Iterable[BrushRecord]) -> None:
        payload = {
            "version": CATALOG_VERSION,
            "maxActiveBrushes": MAX_ACTIVE_BRUSHES,
            "records": [asdict(record) for record in records],
        }
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix="brushes_", suffix=".json.tmp", dir=self.catalog_root)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.catalog_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
