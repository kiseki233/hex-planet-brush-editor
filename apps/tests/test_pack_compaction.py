from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from app.chunk_layout import build_chunk_layout
from app.sphere_map_store import (
    BLOCK_HEADER,
    COMPACTION_BACKUP_DATA_NAME,
    COMPACTION_BACKUP_INDEX_NAME,
    COMPACTION_MARKER_NAME,
    ChunkDataError,
    SphereMapError,
    SphereMapStore,
)
from app.topology import generate_dual_topology


class PackCompactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(8)
        cls.layout = build_chunk_layout(cls.topology, target_cells=32)

    @staticmethod
    def _digest_map(map_dir: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted([map_dir / "index.bin", *((map_dir / "data").glob("pack_*.bin"))]):
            result[path.relative_to(map_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def test_compaction_reclaims_history_and_preserves_every_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            cells = [
                next(cell for cell in chunk.cell_ids if cell >= 12)
                for chunk in self.layout.chunks[: min(6, self.layout.chunk_count)]
            ]
            for generation in range(8):
                for index, cell_id in enumerate(cells):
                    session.set_cell(
                        cell_id,
                        store,
                        f"uid-{index}",
                        f"测试/{index}.png",
                        (generation + index) % 6,
                    )
                store.save(session)

            before = store.analyze_storage(session)
            expected = [store.read_chunk_values(session, chunk_id) for chunk_id in range(self.layout.chunk_count)]
            self.assertGreater(before.orphan_blocks, 0)
            self.assertGreater(before.reclaimable_bytes, 0)

            report = store.compact(session)
            self.assertGreater(report.bytes_reclaimed, 0)
            self.assertEqual(report.after.orphan_blocks, 0)
            self.assertEqual(report.after.reclaimable_bytes, 0)
            self.assertEqual(report.verified_chunks, self.layout.chunk_count)
            self.assertFalse((session.map_dir / COMPACTION_MARKER_NAME).exists())
            self.assertFalse((session.map_dir / COMPACTION_BACKUP_DATA_NAME).exists())
            self.assertFalse((session.map_dir / COMPACTION_BACKUP_INDEX_NAME).exists())

            reopened = store.open("sphere", self.layout)
            actual = [store.read_chunk_values(reopened, chunk_id) for chunk_id in range(self.layout.chunk_count)]
            self.assertEqual(actual, expected)
            self.assertTrue(store.verify(reopened).valid)

    def test_compaction_rejects_unsaved_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            cell_id = next(cell for cell in self.layout.chunks[0].cell_ids if cell >= 12)
            session.set_cell(cell_id, store, "uid", "测试/a.png", 1)
            with self.assertRaises(SphereMapError):
                store.compact(session)
            self.assertTrue(session.dirty_chunks)

    def test_corrupt_orphaned_block_does_not_block_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            chunk_id = 0
            old_record = session.index_records[chunk_id]
            cell_id = next(cell for cell in self.layout.chunks[chunk_id].cell_ids if cell >= 12)
            session.set_cell(cell_id, store, "uid-current", "测试/current.png", 3)
            store.save(session)
            self.assertNotEqual(session.index_records[chunk_id].offset, old_record.offset)
            pack_path = session.map_dir / "data" / f"pack_{old_record.pack_id:04d}.bin"
            with pack_path.open("r+b") as stream:
                stream.seek(old_record.offset + BLOCK_HEADER.size)
                byte = stream.read(1)
                stream.seek(old_record.offset + BLOCK_HEADER.size)
                stream.write(bytes([byte[0] ^ 0xFF]))
            report = store.compact(session)
            self.assertGreater(report.bytes_reclaimed, 0)
            reopened = store.open("sphere", self.layout)
            self.assertEqual(reopened.brush_uid_for_cell(cell_id, store), ("uid-current", 3))
            self.assertTrue(store.verify(reopened).valid)

    def test_live_chunk_corruption_aborts_without_changing_original_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            record = session.index_records[0]
            pack_path = session.map_dir / "data" / f"pack_{record.pack_id:04d}.bin"
            with pack_path.open("r+b") as stream:
                stream.seek(record.offset + BLOCK_HEADER.size)
                byte = stream.read(1)
                stream.seek(record.offset + BLOCK_HEADER.size)
                stream.write(bytes([byte[0] ^ 0xFF]))
            before = self._digest_map(session.map_dir)
            with self.assertRaises(ChunkDataError):
                store.compact(session)
            after = self._digest_map(session.map_dir)
            self.assertEqual(after, before)
            self.assertFalse((session.map_dir / COMPACTION_MARKER_NAME).exists())

    def test_recovery_rolls_back_when_old_data_was_moved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            original_digest = self._digest_map(session.map_dir)
            os.replace(
                session.map_dir / "data",
                session.map_dir / COMPACTION_BACKUP_DATA_NAME,
            )
            store._write_compaction_marker(
                session.map_dir / COMPACTION_MARKER_NAME,
                "old_data_moved",
            )
            report = store.recover_compaction("sphere", self.layout)
            self.assertEqual(report.action, "rolled_back")
            self.assertEqual(report.verified_chunks, self.layout.chunk_count)
            self.assertEqual(self._digest_map(session.map_dir), original_digest)
            self.assertFalse(store.compaction_recovery_state("sphere").required)

    def test_cleanup_failure_keeps_marker_and_can_be_finalized_later(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            cell_id = next(cell for cell in self.layout.chunks[0].cell_ids if cell >= 12)
            session.set_cell(cell_id, store, "uid", "测试/a.png", 1)
            store.save(session)

            original_rmtree = shutil.rmtree

            def fail_backup_cleanup(path: str | Path, *args: object, **kwargs: object) -> None:
                if Path(path).name == COMPACTION_BACKUP_DATA_NAME:
                    raise OSError("simulated cleanup failure")
                original_rmtree(path, *args, **kwargs)

            with patch("app.sphere_map_store.shutil.rmtree", side_effect=fail_backup_cleanup):
                report = store.compact(session)
            self.assertGreater(report.verified_chunks, 0)
            self.assertTrue((session.map_dir / COMPACTION_MARKER_NAME).exists())
            with self.assertRaises(SphereMapError):
                store.open("sphere", self.layout)

            recovery = store.recover_compaction("sphere", self.layout)
            self.assertEqual(recovery.action, "finalized_new")
            self.assertFalse(store.compaction_recovery_state("sphere").required)
            self.assertTrue(store.verify(store.open("sphere", self.layout)).valid)

    def test_failed_rollback_keeps_recovery_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            shutil.rmtree(session.map_dir / "data")
            store._write_compaction_marker(
                session.map_dir / COMPACTION_MARKER_NAME,
                "old_data_moved",
            )
            with self.assertRaises(SphereMapError):
                store.recover_compaction("sphere", self.layout)
            self.assertTrue((session.map_dir / COMPACTION_MARKER_NAME).exists())
            self.assertTrue(store.compaction_recovery_state("sphere").required)

    def test_recovery_finalizes_verified_new_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("sphere", self.layout)
            shutil.copytree(
                session.map_dir / "data",
                session.map_dir / COMPACTION_BACKUP_DATA_NAME,
            )
            shutil.copy2(
                session.map_dir / "index.bin",
                session.map_dir / COMPACTION_BACKUP_INDEX_NAME,
            )
            store._write_compaction_marker(
                session.map_dir / COMPACTION_MARKER_NAME,
                "new_index_installed",
            )
            report = store.recover_compaction("sphere", self.layout)
            self.assertEqual(report.action, "finalized_new")
            self.assertEqual(report.verified_chunks, self.layout.chunk_count)
            self.assertFalse(store.compaction_recovery_state("sphere").required)
            self.assertTrue(store.verify(store.open("sphere", self.layout)).valid)

    def test_recovery_discards_uncommitted_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            stage = session.map_dir / ".pack_compaction_stage"
            (stage / "data").mkdir(parents=True)
            (stage / "data" / "partial.bin").write_bytes(b"partial")
            state = store.compaction_recovery_state("sphere")
            self.assertTrue(state.required)
            self.assertEqual(state.phase, "staging_without_marker")
            report = store.recover_compaction("sphere", self.layout)
            self.assertEqual(report.action, "discarded_stage")
            self.assertFalse(stage.exists())
            self.assertTrue(store.verify(store.open("sphere", self.layout)).valid)

    def test_compaction_refuses_existing_recovery_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps")
            session = store.create_blank("sphere", self.layout)
            store._write_compaction_marker(
                session.map_dir / COMPACTION_MARKER_NAME,
                "prepared",
            )
            with self.assertRaises(SphereMapError):
                store.compact(session)
            with self.assertRaises(SphereMapError):
                store.open("sphere", self.layout)
            recovery = store.recover_compaction("sphere", self.layout)
            self.assertEqual(recovery.action, "rolled_back")


if __name__ == "__main__":
    unittest.main()
