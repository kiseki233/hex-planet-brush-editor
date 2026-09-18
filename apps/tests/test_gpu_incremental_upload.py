from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gpu_batch import GpuInstance
from app.gpu_edit import (
    GpuEditBridge,
    GpuResyncRequest,
    GpuStreamPatch,
    GpuSurfaceRegionPatch,
    GpuSurfaceTexturePatch,
)
from app.gpu_native import _Win32GpuPreview
from app.gpu_stream import GpuInstancePatch
from app.png_pixels import PixelImage


def instance(cell_id: int, layer: int = 0) -> GpuInstance:
    corners = (
        (1.0, 0.0, 0.0),
        (0.8, 0.2, 0.0),
        (0.8, 0.1, 0.2),
        (0.8, -0.1, 0.2),
        (0.8, -0.2, 0.0),
        (0.8, -0.1, -0.2),
    )
    return GpuInstance(cell_id=cell_id, corners=corners, texture_layer=layer, rotation=0)


class GpuIncrementalUploadTests(unittest.TestCase):
    def make_renderer(self):
        renderer = _Win32GpuPreview.__new__(_Win32GpuPreview)
        renderer.instances = []
        renderer.instance_by_cell = {}
        renderer.texture_key_to_layer = {}
        renderer.picker = None
        renderer.selected_instance = -1
        renderer.editable = True
        renderer.last_status = ""
        renderer.view_dirty = False
        renderer.uploaded_slots = []
        renderer._ensure_instance_capacity = lambda required: False
        renderer._upload_instance_slots = lambda slots: renderer.uploaded_slots.append(
            tuple(sorted(int(slot) for slot in slots))
        )
        renderer._ensure_stream_texture = lambda upload: upload.layer
        renderer._update_window_title = lambda: None
        renderer._upload_all_instances = lambda: self.fail(
            "stream patch must not upload the complete instance buffer"
        )
        return renderer

    def test_stream_patch_uploads_only_added_and_swap_moved_slots(self) -> None:
        renderer = self.make_renderer()
        first = instance(10)
        second = instance(20)
        renderer._apply_stream_patch(
            GpuStreamPatch(
                request_id=1,
                active_chunk_ids=(1,),
                instance_count=2,
                removed_cell_ids=(),
                added=(
                    GpuInstancePatch(0, 10, first),
                    GpuInstancePatch(1, 20, second),
                ),
                changed=(),
                texture_uploads=(),
                message="added",
            )
        )
        self.assertEqual(renderer.uploaded_slots[-1], (0, 1))
        self.assertEqual(renderer.instance_by_cell, {10: 0, 20: 1})

        renderer._apply_stream_patch(
            GpuStreamPatch(
                request_id=2,
                active_chunk_ids=(1,),
                instance_count=1,
                removed_cell_ids=(10,),
                added=(),
                changed=(GpuInstancePatch(0, 20, second),),
                texture_uploads=(),
                message="removed",
            )
        )
        self.assertEqual(renderer.uploaded_slots[-1], (0,))
        self.assertEqual([item.cell_id for item in renderer.instances], [20])
        self.assertEqual(renderer.instance_by_cell, {20: 0})

    def test_stream_mismatch_requests_resync_instead_of_closing_gpu(self) -> None:
        renderer = self.make_renderer()
        renderer.edit_bridge = GpuEditBridge()
        renderer.stream_resync_pending = False
        renderer.running = True
        renderer.edit_bridge.push_patch(
            GpuStreamPatch(
                request_id=7,
                active_chunk_ids=(1,),
                instance_count=1,
                removed_cell_ids=(),
                added=(),
                changed=(),
                texture_uploads=(),
                message="stale delta",
            )
        )

        with self.assertLogs(level="ERROR"):
            renderer._poll_edit_bridge()

        self.assertTrue(renderer.running)
        self.assertTrue(renderer.stream_resync_pending)
        requests = renderer.edit_bridge.poll_requests()
        self.assertEqual(len(requests), 1)
        self.assertIsInstance(requests[0], GpuResyncRequest)
        self.assertIn("instance count mismatch", requests[0].reason.lower())

    def test_surface_texture_patch_updates_gpu_surface_without_restarting(self) -> None:
        renderer = _Win32GpuPreview.__new__(_Win32GpuPreview)
        renderer.last_status = ""
        renderer.captured_surface = None
        renderer._upload_surface_texture = lambda image: setattr(
            renderer, "captured_surface", image
        )
        renderer._update_window_title = lambda: None
        pixels = bytes((10, 20, 30)) * 8
        renderer._apply_surface_texture_patch(
            GpuSurfaceTexturePatch(
                request_id=0,
                width=4,
                height=2,
                channels=3,
                pixels=pixels,
                message="updated",
            )
        )
        self.assertIsNotNone(renderer.captured_surface)
        self.assertEqual(renderer.captured_surface.width, 4)
        self.assertEqual(renderer.captured_surface.height, 2)
        self.assertEqual(renderer.captured_surface.pixels, pixels)
        self.assertEqual(renderer.last_status, "updated")

    def test_surface_region_patch_uses_gpu_subimage_upload(self) -> None:
        renderer = _Win32GpuPreview.__new__(_Win32GpuPreview)
        renderer.surface_image = PixelImage(8, 4, 3, bytes((1, 2, 3)) * 32)
        renderer.surface_texture = SimpleNamespace(value=7)
        renderer.last_status = ""
        renderer._update_window_title = lambda: None
        uploads = []
        renderer.gl = {
            "glActiveTexture": lambda *_args: None,
            "glTexSubImage2D": lambda *args: uploads.append(args),
        }
        renderer.opengl32 = SimpleNamespace(
            glBindTexture=lambda *_args: None,
            glPixelStorei=lambda *_args: None,
        )
        pixels = bytes((20, 30, 40)) * 6
        renderer._apply_surface_region_patch(
            GpuSurfaceRegionPatch(
                request_id=0,
                x=2,
                y=1,
                width=3,
                height=2,
                channels=3,
                pixels=pixels,
                message="region updated",
            )
        )
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0][2:6], (2, 1, 3, 2))
        self.assertEqual(renderer.last_status, "region updated")

    def test_small_lod_textures_are_not_limited_to_eight_uploads(self) -> None:
        renderer = _Win32GpuPreview.__new__(_Win32GpuPreview)
        side = 72
        pixels = bytes(side * side * 4)
        renderer.batch = SimpleNamespace(padded_size=side)
        renderer.dragging = False
        renderer.pending_texture_uploads = {
            layer: pixels for layer in range(100)
        }
        renderer.texture_sources = [None] * 100
        renderer.texture_pixels = [pixels] * 100
        renderer.frame_dirty = False
        renderer.last_upload_title_update = 0.0
        uploads = []
        renderer._upload_texture_layer = lambda layer, _pixels: uploads.append(layer)
        renderer._update_window_title = lambda: None

        renderer._flush_pending_texture_uploads()

        self.assertGreater(len(uploads), 8)
        self.assertLess(len(renderer.pending_texture_uploads), 92)


if __name__ == "__main__":
    unittest.main()
