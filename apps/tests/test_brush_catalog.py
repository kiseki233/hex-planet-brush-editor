from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from app.brush_catalog import BrushCatalog, BrushCatalogError
from tests.test_helpers import write_rgb_png


class BrushCatalogTests(unittest.TestCase):
    def test_scan_valid_and_invalid_brushes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            write_rgb_png(root / "城市" / "valid.png")
            write_rgb_png(root / "invalid.png", width=128, height=128)
            result = BrushCatalog(root).scan()
            self.assertEqual(result.active_count, 1)
            self.assertEqual(len(result.invalid), 1)
            self.assertEqual(result.active_records[0].relative_path, "城市/valid.png")
            self.assertEqual(result.active_records[0].average_rgb, (10, 20, 30))

    def test_unchanged_brush_reuses_persisted_average_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            write_rgb_png(root / "城市" / "valid.png", rgb=(12, 34, 56))
            first = BrushCatalog(root).scan().active_records[0]
            self.assertEqual(first.average_rgb, (12, 34, 56))

            with patch(
                "app.brush_catalog.read_png_pixels",
                side_effect=AssertionError("unchanged PNG was decoded again"),
            ), patch(
                "app.brush_catalog.validate_brush_png",
                side_effect=AssertionError("unchanged PNG was validated again"),
            ):
                second = BrushCatalog(root).scan().active_records[0]
            self.assertEqual(second.average_rgb, (12, 34, 56))

    def test_legacy_catalog_without_average_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            write_rgb_png(root / "legacy.png", rgb=(21, 43, 65))
            catalog = BrushCatalog(root)
            first = catalog.scan().active_records[0]
            payload = json.loads(catalog.catalog_path.read_text(encoding="utf-8"))
            payload["records"][0].pop("average_rgb")
            catalog.catalog_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            migrated = catalog.scan().active_records[0]
            self.assertEqual(migrated.uid, first.uid)
            self.assertEqual(migrated.average_rgb, (21, 43, 65))

    def test_same_path_content_change_keeps_uid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            path = root / "ocean.png"
            write_rgb_png(path, rgb=(10, 20, 30))
            first = BrushCatalog(root).scan().active_records[0]
            write_rgb_png(path, rgb=(20, 30, 40))
            second = BrushCatalog(root).scan().active_records[0]
            self.assertEqual(first.uid, second.uid)
            self.assertNotEqual(first.content_hash, second.content_hash)

    def test_unique_move_keeps_uid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            source = root / "old" / "brush.png"
            target = root / "new" / "renamed.png"
            write_rgb_png(source)
            first = BrushCatalog(root).scan().active_records[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            second = BrushCatalog(root).scan().active_records[0]
            self.assertEqual(first.uid, second.uid)
            self.assertEqual(second.relative_path, "new/renamed.png")

    def test_duplicated_content_keeps_one_uid_alive(self) -> None:
        """One brush replaced by two identical copies keeps its uid on one of them.

        The catalog used to decline to pair anything whose content hash was
        ambiguous, which retired the original and minted two fresh uids. That
        turned every cell painted with the original into missing texture even
        though a byte-identical image was still in the library. Since the files
        are identical, whichever copy inherits the uid cannot change what is
        drawn, so preserving it is strictly safer for the map.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            source = root / "old.png"
            write_rgb_png(source)
            first = BrushCatalog(root).scan().active_records[0]
            data = source.read_bytes()
            source.unlink()
            (root / "a.png").write_bytes(data)
            (root / "b.png").write_bytes(data)
            result = BrushCatalog(root).scan()
            active_uids = {record.uid for record in result.active_records}
            self.assertEqual(result.active_count, 2)
            self.assertIn(first.uid, active_uids)
            self.assertEqual(
                [record for record in result.records if record.state == "missing"], []
            )

    def test_extra_identical_copies_still_get_fresh_uids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            source = root / "old.png"
            write_rgb_png(source)
            first = BrushCatalog(root).scan().active_records[0]
            data = source.read_bytes()
            source.unlink()
            for name in ("a.png", "b.png", "c.png"):
                (root / name).write_bytes(data)
            result = BrushCatalog(root).scan()
            uids = {record.uid for record in result.active_records}
            self.assertEqual(result.active_count, 3)
            self.assertIn(first.uid, uids)
            # Only one copy can inherit; the rest are genuinely new brushes.
            self.assertEqual(len(uids), 3)

    def test_deleted_brush_is_marked_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            path = root / "brush.png"
            write_rgb_png(path)
            first = BrushCatalog(root).scan().active_records[0]
            path.unlink()
            result = BrushCatalog(root).scan()
            self.assertEqual(result.active_count, 0)
            self.assertEqual(result.missing_count, 1)
            self.assertEqual(result.records[0].uid, first.uid)

    def test_corrupt_catalog_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "brushes"
            catalog_dir = root / ".catalog"
            catalog_dir.mkdir(parents=True)
            (catalog_dir / "brushes.json").write_text("{broken", encoding="utf-8")
            write_rgb_png(root / "brush.png")
            with self.assertRaises(BrushCatalogError):
                BrushCatalog(root).scan()
            self.assertEqual((catalog_dir / "brushes.json").read_text(encoding="utf-8"), "{broken")


if __name__ == "__main__":
    unittest.main()
