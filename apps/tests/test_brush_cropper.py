from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.brush_cropper_core import (
    BrushCropError,
    SourceTransform,
    TileSelection,
    crop_tile_nearest,
    export_selection,
    numbered_output_paths,
    selection_states,
)
from app.png_pixels import PixelImage, read_png_pixels
from app.paths import ProjectPaths


def split_source(width: int = 1024, height: int = 512) -> PixelImage:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            color = (220, 30, 40) if x < width // 2 else (20, 80, 220)
            pixels[index : index + 3] = bytes(color)
    return PixelImage(width, height, 3, bytes(pixels))


class BrushCropperCoreTests(unittest.TestCase):
    def test_manual_scale_can_fit_1024_source_pixels_into_one_512_tile(self) -> None:
        source = split_source()
        transform = SourceTransform(scale=0.5)
        tile = crop_tile_nearest(source, transform, 0, 0)
        self.assertEqual((tile.width, tile.height, tile.channels), (512, 512, 4))
        left = tile.pixels[(100 * 512 + 64) * 4 : (100 * 512 + 64) * 4 + 4]
        right = tile.pixels[(100 * 512 + 448) * 4 : (100 * 512 + 448) * 4 + 4]
        self.assertEqual(left, bytes((220, 30, 40, 255)))
        self.assertEqual(right, bytes((20, 80, 220, 255)))

    def test_multiple_tiles_preserve_source_regions(self) -> None:
        source = split_source()
        transform = SourceTransform(scale=1.0)
        left = crop_tile_nearest(source, transform, 0, 0)
        right = crop_tile_nearest(source, transform, 1, 0)
        self.assertEqual(left.pixels[:4], bytes((220, 30, 40, 255)))
        self.assertEqual(right.pixels[:4], bytes((20, 80, 220, 255)))

    def test_pan_offset_changes_which_source_region_is_cropped(self) -> None:
        source = split_source()
        transform = SourceTransform(scale=1.0, offset_x=-512.0)
        tile = crop_tile_nearest(source, transform, 0, 0)
        self.assertEqual(tile.pixels[:4], bytes((20, 80, 220, 255)))

    def test_partial_source_area_becomes_transparent(self) -> None:
        source = PixelImage(256, 256, 3, bytes((1, 2, 3)) * (256 * 256))
        tile = crop_tile_nearest(source, SourceTransform(), 0, 0)
        inside = tile.pixels[(100 * 512 + 100) * 4 : (100 * 512 + 100) * 4 + 4]
        outside = tile.pixels[(400 * 512 + 400) * 4 : (400 * 512 + 400) * 4 + 4]
        self.assertEqual(inside, bytes((1, 2, 3, 255)))
        self.assertEqual(outside, bytes((0, 0, 0, 0)))

    def test_selection_reports_full_partial_and_outside_tiles(self) -> None:
        source = split_source()
        states = selection_states(
            source,
            SourceTransform(),
            TileSelection(0, 0, 2, 0),
        )
        self.assertEqual(states, {"full": 2, "partial": 0, "outside": 1})

    def test_export_writes_multiple_512_pngs_with_spatial_names(self) -> None:
        source = split_source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            written = export_selection(
                source,
                SourceTransform(),
                TileSelection(0, 0, 1, 0),
                root,
            )
            self.assertEqual(len(written), 2)
            self.assertEqual(written[0].name, "001.png")
            self.assertEqual(written[1].name, "002.png")
            first = read_png_pixels(written[0])
            second = read_png_pixels(written[1])
            self.assertEqual((first.width, first.height), (512, 512))
            self.assertEqual((second.width, second.height), (512, 512))
            self.assertEqual(first.pixels[:4], bytes((220, 30, 40, 255)))
            self.assertEqual(second.pixels[:4], bytes((20, 80, 220, 255)))

    def test_export_does_not_overwrite_existing_batch(self) -> None:
        source = split_source(512, 512)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            first = export_selection(
                source,
                SourceTransform(),
                TileSelection(),
                root,
            )
            second = export_selection(
                source,
                SourceTransform(),
                TileSelection(),
                root,
            )
            self.assertEqual(first[0].name, "001.png")
            self.assertEqual(second[0].name, "002.png")



    def test_exported_data_receives_uid_only_after_moving_into_brushes(self) -> None:
        source = split_source(512, 512)
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            data_root = project / "art" / "data"
            brush_root = project / "art" / "brushes"
            exported = export_selection(
                source,
                SourceTransform(),
                TileSelection(),
                data_root,
            )
            before = BrushCatalog(brush_root).scan()
            self.assertEqual(before.active_count, 0)

            city_root = brush_root / "城市"
            city_root.mkdir(parents=True)
            target = city_root / exported[0].name
            exported[0].replace(target)

            first = BrushCatalog(brush_root).scan()
            self.assertEqual(first.active_count, 1)
            record = first.active_records[0]
            uuid.UUID(record.uid)
            self.assertEqual(record.relative_path, "城市/001.png")
            original_uid = record.uid

            forest_root = brush_root / "森林"
            forest_root.mkdir(parents=True)
            moved = forest_root / target.name
            target.replace(moved)
            second = BrushCatalog(brush_root).scan()
            self.assertEqual(second.active_count, 1)
            self.assertEqual(second.active_records[0].uid, original_uid)
            self.assertEqual(second.active_records[0].relative_path, "森林/001.png")

    def test_required_directories_include_art_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app_file = root / "apps" / "app" / "main.py"
            app_file.parent.mkdir(parents=True)
            app_file.write_text("", encoding="utf-8")
            paths = ProjectPaths.from_app_file(app_file)
            paths.ensure_required_directories()
            self.assertEqual(paths.data_root, root / "art" / "data")
            self.assertTrue(paths.data_root.is_dir())

    def test_numeric_sequence_ignores_non_numeric_png_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "001.png").write_bytes(b"existing")
            (root / "009.png").write_bytes(b"existing")
            (root / "city.png").write_bytes(b"existing")
            paths = numbered_output_paths(root, 3)
            self.assertEqual([path.name for path in paths], ["010.png", "011.png", "012.png"])

    def test_numbering_expands_beyond_three_digits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "999.png").write_bytes(b"existing")
            paths = numbered_output_paths(root, 2)
            self.assertEqual([path.name for path in paths], ["1000.png", "1001.png"])


if __name__ == "__main__":
    unittest.main()
