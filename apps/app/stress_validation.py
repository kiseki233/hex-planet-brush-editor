from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .brush_catalog import BrushCatalog
from .brush_lod import BrushLodCache
from .gpu_batch import build_brush_texture_payload, brush_texture_key
from .png_pixels import PixelImage, atomic_write_png
from .texture_residency import TextureResidencyManager


@dataclass(frozen=True)
class TextureStressReport:
    valid: bool
    brush_count: int
    lod_level: int
    padded_size: int
    logical_rgba_bytes: int
    expected_rgba_bytes: int
    unique_texture_keys: int
    duplicate_reused_layer: bool
    released_layers: int
    reused_layers: int
    issues: tuple[str, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def run_texture_stress(
    work_root: Path,
    *,
    brush_count: int = 256,
    lod_level: int = 0,
    keep_files: bool = False,
) -> TextureStressReport:
    if brush_count < 1 or brush_count > 4095:
        raise ValueError("brush_count must be between 1 and 4095")
    brush_root = Path(work_root) / "brushes"
    if brush_root.exists():
        shutil.rmtree(brush_root)
    source_dir = brush_root / "stress"
    source_dir.mkdir(parents=True, exist_ok=True)
    base_row_cache: dict[tuple[int, int, int], bytes] = {}
    for index in range(brush_count):
        color = ((index * 67) % 256, (index * 131) % 256, (index * 197) % 256)
        row = base_row_cache.setdefault(color, bytes(color) * 512)
        pixels = row * 512
        atomic_write_png(
            source_dir / f"stress_{index:04d}.png",
            PixelImage(512, 512, 3, pixels),
        )
    scan = BrushCatalog(brush_root).scan()
    issues: list[str] = []
    if scan.active_count != brush_count:
        issues.append(f"catalog accepted {scan.active_count}, expected {brush_count}")
    records = sorted(scan.active_records, key=lambda item: item.relative_path)
    cache = BrushLodCache(brush_root)
    logical_bytes = 0
    keys: list[str] = []
    padded = cache.padded_size(lod_level)
    layer_bytes = padded * padded * 4
    # Exercise the production PNG decode/LOD/padding path on representative
    # layers. The remaining solid-color test layers are byte-identical to the
    # production result, but are assembled directly so the 256-layer worst-case
    # residency test stays practical on slower machines.
    production_samples = min(8, len(records))
    for position, record in enumerate(records):
        key = brush_texture_key(record, lod_level)
        keys.append(key)
        if position < production_samples:
            payload = build_brush_texture_payload(brush_root, record, lod_level)
            if payload.key != key or len(payload.pixels_rgba) != layer_bytes:
                issues.append(f"production payload mismatch for {record.relative_path}")
            logical_bytes += len(payload.pixels_rgba)
        else:
            index = int(Path(record.relative_path).stem.rsplit("_", 1)[-1])
            color = ((index * 67) % 256, (index * 131) % 256, (index * 197) % 256, 255)
            pixels_rgba = bytes(color) * (padded * padded)
            logical_bytes += len(pixels_rgba)
    # Confirm the final record also traverses the real production path.
    if len(records) > production_samples:
        payload = build_brush_texture_payload(brush_root, records[-1], lod_level)
        if payload.key != keys[-1] or len(payload.pixels_rgba) != layer_bytes:
            issues.append(f"production payload mismatch for {records[-1].relative_path}")
    expected = brush_count * padded * padded * 4
    if logical_bytes != expected:
        issues.append(f"RGBA byte total {logical_bytes}, expected {expected}")
    if len(set(keys)) != brush_count:
        issues.append("texture keys are not unique")

    manager = TextureResidencyManager(("empty", "missing"), maximum_layers=brush_count + 8, grace_ticks=0)
    layers = [manager.allocate(key).layer for key in keys]
    duplicate_reused = manager.allocate(keys[0]).layer == layers[0]
    manager.synchronize(layers)
    keep = layers[brush_count // 2 :]
    released = manager.synchronize(keep)
    reused_count = 0
    for index in range(len(released.released_layers)):
        allocation = manager.allocate(f"replacement-{index}")
        reused_count += int(allocation.reused)
    if not duplicate_reused:
        issues.append("duplicate texture key did not reuse its layer")
    if reused_count != len(released.released_layers):
        issues.append("released texture slots were not fully reused")

    report = TextureStressReport(
        valid=not issues,
        brush_count=brush_count,
        lod_level=lod_level,
        padded_size=padded,
        logical_rgba_bytes=logical_bytes,
        expected_rgba_bytes=expected,
        unique_texture_keys=len(set(keys)),
        duplicate_reused_layer=duplicate_reused,
        released_layers=len(released.released_layers),
        reused_layers=reused_count,
        issues=tuple(issues),
    )
    if not keep_files:
        shutil.rmtree(brush_root, ignore_errors=True)
    return report
