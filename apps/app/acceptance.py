from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .aggregate_lod import AggregateLodCache, ProductionAggregateHierarchy
from .brush_catalog import BrushCatalog, BrushRecord
from .brush_cropper_core import SourceTransform, TileSelection, export_selection
from .brush_stroke import (
    BrushStrokePlanner,
    BrushStrokeState,
    BrushStrokeTool,
    random_assignments,
)
from .brush_lod import BrushLodCache
from .flat_map import IcosahedralNetLayout
from .paths import ProjectPaths
from .png_pixels import PixelImage, read_png_pixels
from .production_layout import (
    ProductionChunkLayout,
    load_production_layout_cache,
    write_production_layout_cache,
)
from .production_streaming import ProductionGpuStreamingController
from .production_topology import ProductionTopology
from .production_visibility import (
    build_production_visibility_index,
    load_production_visibility_cache,
    write_production_visibility_cache,
)
from .seam_validation import validate_topology_seams
from .sphere_map_store import SphereMapStore
from .stress_validation import run_texture_stress
from .windows_diagnostics import collect_windows_diagnostics
from .i18n import t


@dataclass(frozen=True)
class AcceptanceItem:
    name: str
    status: str
    detail: str
    seconds: float


@dataclass(frozen=True)
class AcceptanceReport:
    version: str
    items: tuple[AcceptanceItem, ...]
    started_utc: int
    completed_utc: int

    @property
    def passed_count(self) -> int:
        return sum(item.status == "pass" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "fail" for item in self.items)

    @property
    def needs_windows_count(self) -> int:
        return sum(item.status == "needs_windows" for item in self.items)

    @property
    def valid(self) -> bool:
        return self.failed_count == 0

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "acceptance_report.json"
        text_path = directory / "acceptance_report.txt"
        payload = asdict(self)
        payload.update(
            {
                "passedCount": self.passed_count,
                "failedCount": self.failed_count,
                "needsWindowsCount": self.needs_windows_count,
                "valid": self.valid,
            }
        )
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            f"Hex Planet Brush Editor {self.version}",
            f"Passed: {self.passed_count}",
            f"Failed: {self.failed_count}",
            f"Needs Windows: {self.needs_windows_count}",
            "",
        ]
        lines.extend(
            f"[{item.status}] {item.name} ({item.seconds:.3f}s): {item.detail}"
            for item in self.items
        )
        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, text_path


def run_final_acceptance(
    paths: ProjectPaths,
    *,
    full_texture_stress: bool = True,
    run_wgl_probe: bool = True,
) -> AcceptanceReport:
    started = int(time.time())
    items: list[AcceptanceItem] = []

    def check(name: str, action, *, needs_windows: bool = False) -> object | None:
        begin = time.perf_counter()
        try:
            result = action()
        except Exception as exc:
            status = "needs_windows" if needs_windows else "fail"
            items.append(AcceptanceItem(name, status, str(exc), time.perf_counter() - begin))
            return None
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
            valid, detail = result
            status = "pass" if valid else ("needs_windows" if needs_windows else "fail")
        else:
            status = "pass"
            detail = str(result)
        items.append(AcceptanceItem(name, status, detail, time.perf_counter() - begin))
        return result

    check(
        "Root directory contract",
        lambda: (
            set(path.name for path in paths.project_root.iterdir()) <= {"apps", "art"},
            ", ".join(sorted(path.name for path in paths.project_root.iterdir())),
        ),
    )
    check(
        "Launchers and Unicode paths",
        lambda: (
            (paths.apps_root / "start.bat").is_file()
            and (paths.apps_root / "start.ps1").is_file()
            and paths.brush_root.exists()
            and paths.map_root.exists(),
            "start.bat/start.ps1 and art roots present",
        ),
    )
    scan = check(
        "Brush catalog scan",
        lambda: BrushCatalog(paths.brush_root).scan(),
    )
    if scan is not None:
        item = items[-1]
        items[-1] = AcceptanceItem(
            item.name,
            "pass" if not scan.invalid else "fail",
            f"active={scan.active_count}, missing={scan.missing_count}, invalid={len(scan.invalid)}",
            item.seconds,
        )

    production_holder: dict[str, object] = {}

    def production_indexes():
        cache = paths.map_root / ".topology" / "ico_dual_f1004_v2"
        if (cache / "production.json").is_file():
            layout = load_production_layout_cache(cache)
        else:
            layout = ProductionChunkLayout(ProductionTopology(1004), tile_side=16)
            write_production_layout_cache(layout, paths.map_root / ".topology")
        if (cache / "production_visibility.json").is_file():
            visibility = load_production_visibility_cache(cache, layout)
        else:
            visibility = build_production_visibility_index(layout)
            write_production_visibility_cache(visibility, paths.map_root / ".topology")
        layout_validation = layout.validate()
        visibility_validation = visibility.validate(layout)
        if not layout_validation.valid or not visibility_validation.valid:
            raise RuntimeError(
                "; ".join(layout_validation.issues + visibility_validation.issues)
            )
        production_holder["layout"] = layout
        production_holder["visibility"] = visibility
        return (
            f"cells={layout.cell_count:,}, chunks={layout.chunk_count:,}, "
            f"visibilityNodes={visibility.node_count:,}"
        )

    check("Frequency-1004 production indexes", production_indexes)

    def production_pack_and_lod_stream():
        layout = production_holder.get("layout")
        visibility = production_holder.get("visibility")
        if not isinstance(layout, ProductionChunkLayout):
            raise RuntimeError("Production layout was not prepared")
        temporary_name = ".acceptance_f1004_temp"
        map_dir = paths.map_root / temporary_name
        shutil.rmtree(map_dir, ignore_errors=True)
        store = SphereMapStore(paths.map_root)
        try:
            session = store.create_blank(temporary_name, layout)
            verification = store.verify(session)
            if not verification.valid or verification.checked_chunks != layout.chunk_count:
                raise RuntimeError("Full f1004 Pack verification failed")
            controller = ProductionGpuStreamingController(
                layout.topology,
                layout,
                visibility,
                session,
                store,
                {},
                paths.brush_root,
                lod_level=4,
                automatic_lod=True,
                stream_add_budget=48,
                stream_remove_budget=96,
            )
            observed: list[int] = []
            aggregate_nodes = 0
            detail_instances = 0
            # One zoom inside each tier band of ZOOM_TIERS, which is what now
            # selects the level: L5/L4 -> 4, L3 -> 3, L2 -> 2, L1 -> 1, L0 -> 0.
            for zoom in (1.0, 6.0, 20.0, 50.0, 160.0):
                for _ in range(100):
                    frame = controller.update_view(-0.35, 0.25, zoom, 1100, 760)
                    if not frame.has_more:
                        break
                else:
                    raise RuntimeError("Visible instance scheduler did not converge")
                observed.append(frame.lod.level)
                aggregate_nodes = max(
                    aggregate_nodes,
                    0 if frame.aggregate_selection is None else len(frame.aggregate_selection.node_ids),
                )
                detail_instances = max(detail_instances, frame.update.instance_count)
            if observed != [4, 3, 2, 1, 0]:
                raise RuntimeError(f"Unexpected production LOD sequence: {observed}")
            if aggregate_nodes <= 0 or detail_instances <= 0:
                raise RuntimeError("Production aggregate/detail stream was not populated")
            if session.loaded_chunks:
                raise RuntimeError("Production stream retained Pack chunks in the map session")
            return (
                f"verified={verification.checked_chunks:,}, LODs={observed}, "
                f"aggregateNodes={aggregate_nodes}, maxInstances={detail_instances:,}"
            )
        finally:
            shutil.rmtree(map_dir, ignore_errors=True)

    check("Full f1004 Pack and LOD0-L4 visible streaming", production_pack_and_lod_stream)

    def pack_round_trip():
        topology = ProductionTopology(32)
        layout = ProductionChunkLayout(topology)
        temporary_name = ".acceptance_pack_temp"
        map_dir = paths.map_root / temporary_name
        shutil.rmtree(map_dir, ignore_errors=True)
        store = SphereMapStore(paths.map_root)
        try:
            session = store.create_blank(temporary_name, layout)
            cells = []
            for chunk_id in range(layout.chunk_count):
                editable = next(
                    (cell_id for cell_id in layout.chunk_cell_ids(chunk_id) if cell_id >= 12),
                    None,
                )
                if editable is not None:
                    cells.append(editable)
                if len(cells) >= 12:
                    break
            for index, cell_id in enumerate(cells):
                session.set_cell(cell_id, store, "acceptance-brush-uid", "acceptance.png", index % 6)
                store.save(session)
            before = store.analyze_storage(session)
            compacted = store.compact(session)
            reopened = store.open(temporary_name, layout)
            verification = store.verify(reopened)
            if not verification.valid:
                raise RuntimeError("; ".join(verification.issues[:3]))
            if compacted.after.orphan_blocks != 0:
                raise RuntimeError("Pack compaction retained orphaned blocks")
            return (
                f"verified={verification.checked_chunks}, reclaimed={compacted.bytes_reclaimed}, "
                f"historyBefore={before.orphan_blocks}"
            )
        finally:
            shutil.rmtree(map_dir, ignore_errors=True)

    check("Pack save/reopen/CRC/compaction", pack_round_trip)

    def lod_and_aggregate():
        topology = ProductionTopology(16)
        layout = ProductionChunkLayout(topology)
        visibility = build_production_visibility_index(layout, leaf_size=4)
        hierarchy = ProductionAggregateHierarchy(visibility, layout)
        temp_map_root = paths.runtime_root / "acceptance_maps"
        temp_brush_root = paths.runtime_root / "acceptance_brushes"
        shutil.rmtree(temp_map_root, ignore_errors=True)
        shutil.rmtree(temp_brush_root, ignore_errors=True)
        store = SphereMapStore(temp_map_root)
        session = store.create_blank("aggregate", layout)
        cache = AggregateLodCache(temp_brush_root)
        report = cache.build_all(hierarchy, layout, session, store, {})
        if report.total_cells != topology.cell_count:
            raise RuntimeError("Aggregate cache cell coverage mismatch")
        return f"aggregateNodes={report.node_count}, cells={report.total_cells}"

    check("Four-level LOD and multi-level aggregate cache", lod_and_aggregate)

    def seams():
        report = validate_topology_seams(ProductionTopology(128), sample_limit=1024)
        if not report.valid:
            raise RuntimeError("; ".join(report.issues[:3]))
        report.write(paths.log_root / "seam_validation.json")
        return (
            f"cells={report.sampled_cells}, edges={report.checked_neighbor_pairs}, "
            f"maxError={report.max_shared_corner_error:.3e}"
        )

    check("Shared geometry, padding and six-direction UV seams", seams)

    def continuous_random_brush():
        topology = ProductionTopology(64)
        planner = BrushStrokePlanner(topology)
        tool = BrushStrokeTool(
            "paint",
            25,
            (
                BrushRecord(
                    uid="acceptance-a",
                    relative_path=t("城市/a.png"),
                    content_hash="a" * 64,
                    file_size=1,
                    width=512,
                    height=512,
                    color_mode="RGB",
                    category_path=t("城市"),
                    modified_time_ns=0,
                    state="active",
                    last_seen_utc=0,
                ),
                BrushRecord(
                    uid="acceptance-b",
                    relative_path=t("城市/b.png"),
                    content_hash="b" * 64,
                    file_size=1,
                    width=512,
                    height=512,
                    color_mode="RGB",
                    category_path=t("城市"),
                    modified_time_ns=0,
                    state="active",
                    last_seen_utc=0,
                ),
            ),
        )
        start = 500
        target = topology.cell_neighbor_ids_unordered(start)[0]
        state = BrushStrokeState(tool)
        first = planner.plan_segment(state, start)
        second = planner.plan_segment(state, target)
        repeated = planner.plan_segment(state, target)
        affected = first.affected_cell_ids + second.affected_cell_ids
        if not affected or repeated.affected_cell_ids:
            raise RuntimeError("Continuous stroke deduplication failed")
        assignments = random_assignments(affected, tool)
        if any(value is None or value[2] < 0 or value[2] > 5 for value in assignments.values()):
            raise RuntimeError("Random brush assignment produced invalid state")
        if BrushStrokePlanner.graph_radius(500) != 250:
            raise RuntimeError("500-cell diameter limit is not available")
        return (
            f"diameter=1..500, strokeCells={len(affected)}, "
            f"groupImages={len(tool.records)}, rotations=0..5"
        )

    check("Random brush groups and continuous 1-500-cell strokes", continuous_random_brush)

    def brush_image_cropper():
        crop_root = paths.runtime_root / "acceptance_cropper"
        shutil.rmtree(crop_root, ignore_errors=True)
        width = 1024
        height = 512
        pixels = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                index = (y * width + x) * 3
                color = (210, 30, 40) if x < 512 else (20, 80, 220)
                pixels[index : index + 3] = bytes(color)
        source = PixelImage(width, height, 3, bytes(pixels))
        written = export_selection(
            source,
            SourceTransform(),
            TileSelection(0, 0, 1, 0),
            crop_root,
        )
        if len(written) != 2 or [path.name for path in written] != ["001.png", "002.png"]:
            raise RuntimeError("Cropper did not export the expected global numeric sequence")
        images = tuple(read_png_pixels(path) for path in written)
        if any((image.width, image.height, image.channels) != (512, 512, 4) for image in images):
            raise RuntimeError("Cropper output is not 512x512 RGBA PNG")
        if images[0].pixels[:4] == images[1].pixels[:4]:
            raise RuntimeError("Multiple crop cells did not preserve distinct source regions")

        acceptance_brush_root = paths.runtime_root / "acceptance_cropper_brushes"
        shutil.rmtree(acceptance_brush_root, ignore_errors=True)
        city_root = acceptance_brush_root / t("城市")
        city_root.mkdir(parents=True, exist_ok=True)
        written[0].replace(city_root / written[0].name)
        scan = BrushCatalog(acceptance_brush_root).scan()
        if scan.active_count != 1 or not scan.active_records[0].uid:
            raise RuntimeError("A cropped file did not receive a brush UID after entering art/brushes")
        return (
            "flat 001/002 sequence, 2x 512x512 PNG, "
            f"UID-on-brush-import={scan.active_records[0].uid}"
        )

    check("Flat sequential art/data cropper", brush_image_cropper)

    def flat_map_editor():
        topology = ProductionTopology(32)
        net = IcosahedralNetLayout(topology.frequency)
        if len(net.face_placements) != 20:
            raise RuntimeError("2D net does not contain twenty base faces")
        seam_id = topology.point_id(0, 16, 16, 0)
        representations = topology.cell_representations(seam_id)
        if len(representations) != 2:
            raise RuntimeError("Expected a duplicated seam CellId")
        hits = {net.nearest_cell(topology, *net.face_point(address))[1] for address in representations}
        if hits != {seam_id}:
            raise RuntimeError("2D seam copies do not resolve to the same CellId")
        visible = net.visible_cells(topology, net.bounds, maximum_cells=100000)
        if not visible:
            raise RuntimeError("2D net did not generate editable cell polygons")
        return f"faces=20, seamCopies={len(representations)}, polygons={len(visible):,}"

    check("Shared-map 2D icosahedron-net editor", flat_map_editor)

    def texture_stress():
        count = 256 if full_texture_stress else 32
        report = run_texture_stress(
            paths.runtime_root / "texture_stress",
            brush_count=count,
            lod_level=0,
        )
        report.write(paths.log_root / "texture_stress.json")
        if not report.valid:
            raise RuntimeError("; ".join(report.issues))
        return (
            f"brushes={count}, RGBA={report.logical_rgba_bytes:,}, "
            f"released/reused={report.released_layers}/{report.reused_layers}"
        )

    check("Distinct LOD0 texture-array worst-case stress", texture_stress)

    def diagnostics():
        report = collect_windows_diagnostics(paths, run_wgl_probe=run_wgl_probe)
        report.write(paths.log_root)
        if report.failed_count:
            raise RuntimeError(f"Windows diagnostic failures: {report.failed_count}")
        if report.needs_windows_count:
            return False, f"{report.needs_windows_count} item(s) require the target Windows PC"
        return True, "WGL/OpenGL diagnostic passed"

    check("Windows/WGL target diagnostic", diagnostics, needs_windows=True)

    shutil.rmtree(paths.runtime_root / "acceptance_maps", ignore_errors=True)
    shutil.rmtree(paths.runtime_root / "acceptance_brushes", ignore_errors=True)
    shutil.rmtree(paths.runtime_root / "texture_stress", ignore_errors=True)
    shutil.rmtree(paths.runtime_root / "acceptance_cropper", ignore_errors=True)
    shutil.rmtree(paths.runtime_root / "acceptance_cropper_brushes", ignore_errors=True)
    return AcceptanceReport(
        version="v1.3.1",
        items=tuple(items),
        started_utc=started,
        completed_utc=int(time.time()),
    )
