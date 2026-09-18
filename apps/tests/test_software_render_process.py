from __future__ import annotations

import time
import unittest

from app.png_pixels import PixelImage
from app.software_render_process import (
    SoftwareGlobeRenderProcess,
    SoftwareRenderRequest,
)


class SoftwareRenderProcessTests(unittest.TestCase):
    def test_process_keeps_latest_request_and_returns_ppm(self) -> None:
        texture = PixelImage(
            8,
            4,
            3,
            bytes((80, 120, 160)) * 32,
        )
        renderer = SoftwareGlobeRenderProcess()
        try:
            renderer.set_texture(texture)
            renderer.submit(
                SoftwareRenderRequest(
                    generation=1,
                    key=("first",),
                    yaw=0.0,
                    pitch=0.0,
                    width=80,
                    height=60,
                    center_x=40.0,
                    center_y=30.0,
                    radius=24.0,
                    block_size=4,
                    preview_scale=4,
                )
            )
            renderer.submit(
                SoftwareRenderRequest(
                    generation=2,
                    key=("latest",),
                    yaw=0.4,
                    pitch=-0.2,
                    width=80,
                    height=60,
                    center_x=40.0,
                    center_y=30.0,
                    radius=24.0,
                    block_size=4,
                    preview_scale=4,
                )
            )
            deadline = time.monotonic() + 8.0
            latest = None
            while time.monotonic() < deadline:
                result = renderer.poll_latest()
                if result is not None:
                    latest = result
                    if result.generation == 2:
                        break
                time.sleep(0.02)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.generation, 2)
            self.assertIsNone(latest.error)
            self.assertTrue(latest.ppm.startswith(b"P6\n80 60\n255\n"))
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
