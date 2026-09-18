from __future__ import annotations

import ctypes
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.brush_catalog import BrushCatalog
from app.chunk_layout import build_chunk_layout
from app.gpu_batch import (
    INSTANCE_STRUCT,
    EMPTY_LAYER_KEY,
    GpuRenderBatchBuilder,
    brush_texture_key,
    build_brush_texture_payload,
)
from app.gpu_edit import (
    GpuCellPatch,
    GpuEditBridge,
    GpuEditRequest,
    GpuResyncRequest,
    GpuSaveRequest,
    GpuStatusPatch,
    GpuToolState,
    SphericalCellPicker,
    screen_to_world_direction,
)
from app.gpu_native import FRAGMENT_SHADER_SOURCE, VERTEX_SHADER_SOURCE, _Win32GpuPreview
from app.sphere_editor import SphereMapEditor
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology
from tests.test_helpers import write_rgb_png


class GpuEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(8)
        cls.layout = build_chunk_layout(cls.topology, target_cells=64)

    def _blank_batch(self, root: Path):
        store = SphereMapStore(root / "maps")
        session = store.create_blank("sphere", self.layout)
        batch = GpuRenderBatchBuilder(root / "brushes").build(
            self.topology,
            self.layout,
            session,
            store,
            {},
            range(self.topology.cell_count),
            3,
        )
        return store, session, batch

    def test_bridge_snapshots_tool_state_and_routes_requests_and_patches(self) -> None:
        bridge = GpuEditBridge(
            GpuToolState(
                "paint", "uid-a", "地形/a.png", 2,
                brush_group="地形", brush_diameter=37,
            ),
            known_texture_keys=(EMPTY_LAYER_KEY,),
        )
        edit_id = bridge.submit_edit(42, stroke_id=9, phase="start")
        save_id = bridge.submit_save()
        resync_id = bridge.submit_resync("instance mismatch")
        requests = bridge.poll_requests()
        self.assertEqual(len(requests), 3)
        self.assertIsInstance(requests[0], GpuEditRequest)
        self.assertEqual(requests[0].request_id, edit_id)
        self.assertEqual(requests[0].state.brush_uid, "uid-a")
        self.assertEqual(requests[0].state.rotation, 2)
        self.assertEqual(requests[0].state.brush_group, "地形")
        self.assertEqual(requests[0].state.brush_diameter, 37)
        self.assertEqual(requests[0].stroke_id, 9)
        self.assertEqual(requests[0].phase, "start")
        self.assertIsInstance(requests[1], GpuSaveRequest)
        self.assertEqual(requests[1].request_id, save_id)
        self.assertIsInstance(requests[2], GpuResyncRequest)
        self.assertEqual(requests[2].request_id, resync_id)
        self.assertEqual(requests[2].reason, "instance mismatch")

        bridge.set_tool("erase")
        bridge.cycle_rotation()
        next_id = bridge.submit_edit(43, stroke_id=9, phase="move")
        end_id = bridge.submit_stroke_end(9)
        next_request, end_request = bridge.poll_requests()
        self.assertEqual(next_request.request_id, next_id)
        self.assertEqual(next_request.state.tool, "erase")
        self.assertEqual(next_request.state.rotation, 3)
        self.assertEqual(next_request.state.brush_diameter, 37)
        self.assertEqual(end_request.request_id, end_id)
        self.assertEqual(end_request.phase, "end")
        self.assertEqual(end_request.cell_id, -1)

        patch = GpuCellPatch(edit_id, 42, EMPTY_LAYER_KEY, 0, "ok")
        status = GpuStatusPatch(save_id, True, "saved")
        bridge.push_patch(patch)
        bridge.push_patch(status)
        self.assertEqual(bridge.poll_patches(), (patch, status))

    def test_native_continuous_stroke_submits_start_and_deduplicated_moves(self) -> None:
        class Picker:
            topology = SimpleNamespace(pentagon_ids=())

            def __init__(self):
                self.cell_id = 42

            def pick_screen(self, *args):
                return SimpleNamespace(
                    cell_id=self.cell_id, instance_index=0, world_direction=(0.0, 0.0, 1.0)
                )

        native = object.__new__(_Win32GpuPreview)
        native.picker = Picker()
        native.edit_bridge = GpuEditBridge(
            GpuToolState(
                "paint", "uid-a", "城市/a.png", 0,
                brush_group="城市", brush_diameter=25,
            )
        )
        native.editable = True
        native.selected_instance = -1
        native.last_status = ""
        native.last_paint_cell = None
        native.width = 100
        native.height = 100
        native.yaw = 0.0
        native.pitch = 0.0
        native.zoom = 1.0
        native.hwnd = None
        native._update_window_title = lambda: None

        self.assertEqual(native._submit_edit_at(50, 50, phase="start", stroke_id=7), 42)
        self.assertEqual(native._submit_edit_at(51, 50, phase="move", stroke_id=7), 42)
        native.picker.cell_id = 43
        self.assertEqual(native._submit_edit_at(52, 50, phase="move", stroke_id=7), 43)
        requests = native.edit_bridge.poll_requests()
        self.assertEqual(len(requests), 2)
        self.assertEqual((requests[0].phase, requests[0].stroke_id), ("start", 7))
        self.assertEqual((requests[1].phase, requests[1].stroke_id), ("move", 7))
        self.assertEqual(requests[1].state.brush_diameter, 25)

    def test_bridge_texture_claim_is_exactly_once_until_forgotten(self) -> None:
        bridge = GpuEditBridge(known_texture_keys=("existing",))
        self.assertFalse(bridge.claim_texture_key("existing"))
        self.assertTrue(bridge.claim_texture_key("new"))
        self.assertFalse(bridge.claim_texture_key("new"))
        bridge.forget_texture_key("new")
        self.assertTrue(bridge.claim_texture_key("new"))

    def test_screen_direction_round_trip_and_outside_rejection(self) -> None:
        direction = screen_to_world_direction(550, 380, 1100, 760, 0.7, -0.35, 2.0)
        self.assertIsNotNone(direction)
        assert direction is not None
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in direction)), 1.0, places=7)
        expected = (-math.sin(0.7) * math.cos(-0.35), math.sin(-0.35), math.cos(0.7) * math.cos(-0.35))
        for actual, target in zip(direction, expected):
            self.assertAlmostEqual(actual, target, places=7)
        self.assertIsNone(screen_to_world_direction(-5000, -5000, 1100, 760, 0.0, 0.0, 1.0))

    def test_topology_picker_matches_brute_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, batch = self._blank_batch(Path(temporary))
            picker = SphericalCellPicker(self.topology, batch)
            randomizer = random.Random(20260726)
            for _ in range(80):
                vector = (
                    randomizer.uniform(-1.0, 1.0),
                    randomizer.uniform(-1.0, 1.0),
                    randomizer.uniform(-1.0, 1.0),
                )
                length = math.sqrt(sum(value * value for value in vector))
                direction = tuple(value / length for value in vector)
                picked = picker.nearest_cell(direction)
                brute = max(
                    range(self.topology.cell_count),
                    key=lambda cell_id: sum(
                        a * b for a, b in zip(self.topology.cell_centers[cell_id], direction)
                    ),
                )
                self.assertEqual(picked, brute)

    def test_screen_picker_returns_instance_for_editable_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, batch = self._blank_batch(Path(temporary))
            picker = SphericalCellPicker(self.topology, batch)
            result = picker.pick_screen(550, 380, 1100, 760, 0.23, -0.18, 1.6)
            self.assertIsNotNone(result)
            assert result is not None
            if result.cell_id in self.topology.pentagon_ids:
                self.assertIsNone(result.instance_index)
            else:
                self.assertIsNotNone(result.instance_index)
                assert result.instance_index is not None
                self.assertEqual(batch.instances[result.instance_index].cell_id, result.cell_id)

    def test_brush_texture_payload_matches_batch_key_and_lod_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root = Path(temporary) / "brushes"
            write_rgb_png(brush_root / "地形" / "green.png", rgb=(20, 140, 65))
            record = BrushCatalog(brush_root).scan().active_records[0]
            payload = build_brush_texture_payload(brush_root, record, 2)
            self.assertEqual(payload.key, brush_texture_key(record, 2))
            self.assertEqual((payload.width, payload.height), (136, 136))
            self.assertEqual(len(payload.pixels_rgba), 136 * 136 * 4)


    def test_authoritative_gpu_edit_handler_round_trips_through_pack(self) -> None:
        class Loader:
            def ensure_loaded(self, session, chunk_id):
                return session.loaded_chunks.get(chunk_id) or store.load_chunk(session, chunk_id)

        class Distant:
            def __init__(self):
                self.invalidated = []

            def invalidate(self, chunk_ids):
                self.invalidated.extend(chunk_ids)

        class Status:
            def __init__(self):
                self.value = ""

            def set(self, value):
                self.value = value

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            brush_root = root / "brushes"
            write_rgb_png(brush_root / "地形" / "green.png", rgb=(30, 170, 90))
            record = BrushCatalog(brush_root).scan().active_records[0]
            topology = generate_dual_topology(4)
            layout = build_chunk_layout(topology, target_cells=32)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("sphere", layout)

            editor = object.__new__(SphereMapEditor)
            editor.session = session
            editor.topology = topology
            editor.gpu_edit_lod_level = 2
            editor.chunk_loader = Loader()
            editor.store = store
            editor.records_by_uid = {record.uid: record}
            editor.paths = SimpleNamespace(brush_root=brush_root)
            editor.selected_cell_id = None
            editor.distant_builder = Distant()
            editor.status = Status()

            bridge = GpuEditBridge(known_texture_keys=(EMPTY_LAYER_KEY, "__missing__"))
            request = GpuEditRequest(1, 12, GpuToolState("paint", record.uid, record.relative_path, 4))
            self.assertTrue(editor._handle_gpu_edit_request(bridge, request))
            patch = bridge.poll_patches()[0]
            self.assertIsInstance(patch, GpuCellPatch)
            self.assertEqual(patch.rotation, 4)
            self.assertIsNotNone(patch.pixels_rgba)
            self.assertTrue(session.dirty_chunks)

            editor.projection = None
            editor.current_lod_level = 2
            editor.planet_refresh_requested = False
            editor._request_planet_cache = lambda: None
            save_request = GpuSaveRequest(2)
            editor._handle_gpu_save_request(bridge, save_request)
            save_patch = bridge.poll_patches()[0]
            self.assertIsInstance(save_patch, GpuStatusPatch)
            self.assertTrue(save_patch.success)

            reopened = store.open("sphere", layout)
            uid, rotation = reopened.brush_uid_for_cell(12, store)
            self.assertEqual(uid, record.uid)
            self.assertEqual(rotation, 4)


    def test_native_mutable_patch_updates_one_instance_and_reuses_texture_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, batch = self._blank_batch(Path(temporary))
            native = object.__new__(_Win32GpuPreview)
            native.batch = batch
            native.instances = list(batch.instances)
            native.instance_by_cell = {
                instance.cell_id: index for index, instance in enumerate(batch.instances)
            }
            side = batch.padded_size
            native.texture_pixels = [bytes(side * side * 4), bytes(side * side * 4)]
            native.texture_key_to_layer = {"__empty__": 0, "__missing__": 1}
            native.maximum_texture_layers = 64
            native.texture_capacity = 32
            native.selected_instance = -1
            native.last_status = ""
            native.hwnd = None
            native.buffers = (ctypes.c_uint * 2)(1, 2)
            calls = []
            native.gl = {
                "glBindBuffer": lambda *args: calls.append(("bind", args)),
                "glBufferSubData": lambda *args: calls.append(("sub", args[:3])),
            }
            uploads = []
            native._upload_texture_layer = lambda layer, pixels: uploads.append((layer, len(pixels)))
            native._allocate_texture_storage = lambda: None

            cell_id = batch.instances[0].cell_id
            pixels = bytes([7, 8, 9, 255]) * (side * side)
            patch = GpuCellPatch(1, cell_id, "brush:new:lod3", 5, "painted", pixels, side, side)
            native._apply_cell_patch(patch)
            index = native.instance_by_cell[cell_id]
            self.assertEqual(native.instances[index].rotation, 5)
            self.assertEqual(native.instances[index].texture_layer, 2)
            self.assertEqual(uploads, [(2, len(pixels))])
            self.assertTrue(any(call[0] == "sub" and call[1][1] == index * INSTANCE_STRUCT.size for call in calls))

            second = GpuCellPatch(2, cell_id, "brush:new:lod3", 2, "repainted")
            native._apply_cell_patch(second)
            self.assertEqual(native.instances[index].rotation, 2)
            self.assertEqual(native.instances[index].texture_layer, 2)
            self.assertEqual(len(uploads), 1)

    def test_shader_contract_includes_selection_and_mutable_instance_id(self) -> None:
        self.assertIn("gl_InstanceID", VERTEX_SHADER_SOURCE)
        self.assertIn("uSelectedInstance", VERTEX_SHADER_SOURCE + FRAGMENT_SHADER_SOURCE)
        self.assertIn("vInstanceId == uSelectedInstance", FRAGMENT_SHADER_SOURCE)


if __name__ == "__main__":
    unittest.main()
