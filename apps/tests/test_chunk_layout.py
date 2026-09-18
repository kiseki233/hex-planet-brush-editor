from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.chunk_layout import (
    ChunkLayoutError,
    build_chunk_layout,
    load_chunk_layout_cache,
    write_chunk_layout_cache,
)
from app.topology import generate_dual_topology


class ChunkLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(16)

    def test_layout_is_stable_connected_and_complete(self) -> None:
        first = build_chunk_layout(self.topology, target_cells=256)
        second = build_chunk_layout(self.topology, target_cells=256)
        validation = first.validate(self.topology)
        self.assertTrue(validation.valid, validation.issues)
        self.assertTrue(validation.connected_chunks)
        self.assertEqual(first.stable_hash, second.stable_hash)
        self.assertEqual(first.cell_count, self.topology.cell_count)
        self.assertEqual(sum(len(chunk.cell_ids) for chunk in first.chunks), first.cell_count)
        self.assertLessEqual(max(len(chunk.cell_ids) for chunk in first.chunks), 256)
        self.assertEqual({chunk.base_face for chunk in first.chunks}, set(range(20)))

    def test_small_remainder_chunks_are_rebalanced(self) -> None:
        topology = generate_dual_topology(32)
        layout = build_chunk_layout(topology, target_cells=256)
        sizes = [len(chunk.cell_ids) for chunk in layout.chunks]
        self.assertGreaterEqual(min(sizes), 128)
        self.assertLessEqual(max(sizes), 256)
        self.assertTrue(layout.validate(topology).connected_chunks)

    def test_cell_mapping_round_trip(self) -> None:
        layout = build_chunk_layout(self.topology)
        for chunk in layout.chunks:
            for local_index, cell_id in enumerate(chunk.cell_ids):
                self.assertEqual(layout.chunk_for_cell(cell_id), (chunk.chunk_id, local_index))

    def test_cache_round_trip(self) -> None:
        layout = build_chunk_layout(self.topology)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_chunk_layout_cache(layout, Path(temporary) / ".topology")
            loaded = load_chunk_layout_cache(info.directory)
            self.assertEqual(loaded.stable_hash, layout.stable_hash)
            self.assertEqual(loaded.cell_to_chunk, layout.cell_to_chunk)
            self.assertEqual(loaded.chunks, layout.chunks)

    def test_corrupt_cache_is_reported(self) -> None:
        layout = build_chunk_layout(self.topology)
        with tempfile.TemporaryDirectory() as temporary:
            info = write_chunk_layout_cache(layout, Path(temporary) / ".topology")
            payload = bytearray(info.index_path.read_bytes())
            payload[-1] ^= 0xFF
            info.index_path.write_bytes(payload)
            with self.assertRaises(ChunkLayoutError):
                load_chunk_layout_cache(info.directory)


if __name__ == "__main__":
    unittest.main()
