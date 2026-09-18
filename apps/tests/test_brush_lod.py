from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.brush_lod import BrushLodCache, BrushLodPolicy, LOD_EFFECTIVE_SIZES
from app.png_pixels import PixelImage, atomic_write_png, read_png_pixels


class BrushLodTests(unittest.TestCase):
    def test_four_lod_levels_have_edge_padding_and_stable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            source = root / "地形" / "gradient.png"
            source.parent.mkdir(parents=True)
            pixels = bytearray(512 * 512 * 3)
            for y in range(512):
                for x in range(512):
                    index = (y * 512 + x) * 3
                    pixels[index : index + 3] = bytes((x & 0xFF, y & 0xFF, (x + y) & 0xFF))
            atomic_write_png(source, PixelImage(512, 512, 3, bytes(pixels)))
            record = BrushCatalog(root).scan().active_records[0]
            cache = BrushLodCache(root)

            infos = cache.ensure_all(record)
            self.assertEqual(tuple(info.effective_size for info in infos), LOD_EFFECTIVE_SIZES)
            self.assertTrue(all(info.generated for info in infos))
            for info in infos:
                image = read_png_pixels(info.path)
                self.assertEqual(image.width, info.effective_size + 8)
                self.assertEqual(image.height, info.effective_size + 8)
                channels = image.channels

                def pixel(x: int, y: int) -> bytes:
                    offset = (y * image.width + x) * channels
                    return image.pixels[offset : offset + channels]

                self.assertEqual(pixel(0, 0), pixel(4, 4))
                self.assertEqual(pixel(image.width - 1, 0), pixel(image.width - 5, 4))
                self.assertEqual(pixel(0, image.height - 1), pixel(4, image.height - 5))
                self.assertEqual(
                    pixel(image.width - 1, image.height - 1),
                    pixel(image.width - 5, image.height - 5),
                )

            second = cache.ensure_all(record)
            self.assertTrue(all(not info.generated for info in second))
            self.assertEqual(tuple(info.path for info in infos), tuple(info.path for info in second))

    def test_content_change_uses_new_hash_directory_and_drops_the_old(self) -> None:
        """Replacing a brush keeps its uid but supersedes its cached levels.

        The superseded directory is deleted rather than left behind: nothing
        refers to it once the record carries the new hash, and across a whole
        library replacement the strays would amount to the entire previous cache.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            source = root / "brush.png"
            first = PixelImage(512, 512, 3, bytes((10, 20, 30)) * (512 * 512))
            atomic_write_png(source, first)
            catalog = BrushCatalog(root)
            record_one = catalog.scan().active_records[0]
            cache = BrushLodCache(root)
            path_one = cache.ensure(record_one, 2).path

            second = PixelImage(512, 512, 3, bytes((30, 20, 10)) * (512 * 512))
            atomic_write_png(source, second)
            record_two = catalog.scan().active_records[0]
            path_two = cache.ensure(record_two, 2).path
            self.assertEqual(record_one.uid, record_two.uid)
            self.assertNotEqual(record_one.content_hash, record_two.content_hash)
            self.assertNotEqual(path_one, path_two)
            self.assertFalse(path_one.exists())
            self.assertFalse(path_one.parent.exists())
            self.assertTrue(path_two.exists())

    def test_lod_policy_uses_hysteresis(self) -> None:
        policy = BrushLodPolicy()
        self.assertEqual(policy.update(256), 0)
        self.assertEqual(policy.update(290), 1)
        self.assertEqual(policy.update(250), 1)
        self.assertEqual(policy.update(224), 0)
        self.assertEqual(policy.update(17000), 4)
        self.assertEqual(policy.update(16200), 4)
        self.assertEqual(policy.update(16000), 3)
        self.assertEqual(policy.update(1000), 2)
        self.assertEqual(policy.update(900), 1)


if __name__ == "__main__":
    unittest.main()
