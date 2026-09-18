from __future__ import annotations

import unittest

from app.gpu_edit import GpuBatchResetPatch, GpuSurfaceTexturePatch, GpuTextureUpload
from app.gpu_native import NativeGpuViewportHandle


class GpuPatchContractTests(unittest.TestCase):
    def test_texture_upload_can_replace_recycled_layer(self) -> None:
        upload = GpuTextureUpload(7, "new-key", 1, 1, b"\x00\x00\x00\xff", True)
        self.assertTrue(upload.replace_existing)
        self.assertEqual(upload.layer, 7)

    def test_batch_reset_patch_exposes_editability_and_followup(self) -> None:
        fields = GpuBatchResetPatch.__dataclass_fields__
        self.assertIn("editable", fields)
        self.assertIn("has_more", fields)

    def test_surface_texture_patch_carries_rgb_or_rgba_payload(self) -> None:
        patch = GpuSurfaceTexturePatch(3, 2, 1, 3, bytes((1, 2, 3, 4, 5, 6)))
        self.assertEqual((patch.width, patch.height, patch.channels), (2, 1, 3))
        self.assertEqual(len(patch.pixels), 6)

    def test_native_viewport_handle_is_safe_without_windows_hwnd(self) -> None:
        handle = NativeGpuViewportHandle()
        handle.resize(640, 480)
        handle.close()
        self.assertIsNone(handle.hwnd)


if __name__ == "__main__":
    unittest.main()
