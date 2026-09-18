from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.chunk_layout import build_chunk_layout
from app.chunk_visibility import (
    ChunkVisibilityError,
    build_chunk_visibility_index,
    load_chunk_visibility_cache,
    verify_chunk_visibility_cache,
    write_chunk_visibility_cache,
)
from app.sphere_viewport import SphereViewport
from app.topology import generate_dual_topology


class ChunkVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(16)
        cls.layout = build_chunk_layout(cls.topology, target_cells=64)
        cls.index = build_chunk_visibility_index(cls.topology, cls.layout, leaf_size=4)

    def test_hierarchy_is_stable_complete_and_valid(self) -> None:
        second = build_chunk_visibility_index(self.topology, self.layout, leaf_size=4)
        validation = self.index.validate(self.topology, self.layout)
        self.assertTrue(validation.valid, validation.issues)
        self.assertEqual(self.index.stable_hash, second.stable_hash)
        self.assertEqual(validation.chunk_count, self.layout.chunk_count)
        self.assertGreater(validation.node_count, 1)
        self.assertGreater(validation.boundary_point_count, self.layout.chunk_count * 3)

    def test_query_is_conservative_for_multiple_views(self) -> None:
        views = (
            (-0.35, 0.25, 1.0),
            (1.2, -0.7, 2.5),
            (2.8, 0.9, 6.0),
            (0.0, 0.0, 0.8),
        )
        for yaw, pitch, zoom in views:
            viewport = SphereViewport(yaw=yaw, pitch=pitch, zoom=zoom)
            full = viewport.project(self.topology, 900, 700, self.layout)
            query = self.index.query(self.layout, yaw, pitch, zoom, 900, 700)
            self.assertTrue(set(full.visible_chunk_ids).issubset(set(query.chunk_ids)))
            self.assertLessEqual(query.visited_nodes, self.index.node_count)
            self.assertLessEqual(query.tested_chunks, self.layout.chunk_count)

    def test_indexed_detail_projection_matches_full_projection(self) -> None:
        viewport = SphereViewport(yaw=0.7, pitch=-0.35, zoom=3.5)
        full = viewport.project(self.topology, 900, 700, self.layout)
        query = self.index.query(
            self.layout, viewport.yaw, viewport.pitch, viewport.zoom, 900, 700
        )
        candidate_cells = self.index.candidate_cell_ids(self.layout, query.chunk_ids)
        indexed = viewport.project(
            self.topology,
            900,
            700,
            self.layout,
            cell_ids=candidate_cells,
            candidate_cell_count=query.candidate_cells,
            visited_visibility_nodes=query.visited_nodes,
            tested_visibility_chunks=query.tested_chunks,
        )
        self.assertEqual(set(full.visible_cell_ids), set(indexed.visible_cell_ids))
        self.assertEqual(set(full.visible_chunk_ids), set(indexed.visible_chunk_ids))
        self.assertLess(indexed.candidate_cell_count, self.topology.cell_count)

    def test_distant_projection_uses_chunk_boundaries_without_cells(self) -> None:
        viewport = SphereViewport(zoom=4.0)
        query = self.index.query(
            self.layout, viewport.yaw, viewport.pitch, viewport.zoom, 900, 700
        )
        projection = viewport.project_chunks(
            self.topology,
            self.index,
            query.chunk_ids,
            900,
            700,
            candidate_cell_count=query.candidate_cells,
            visited_visibility_nodes=query.visited_nodes,
            tested_visibility_chunks=query.tested_chunks,
        )
        self.assertEqual(projection.cells, ())
        self.assertEqual(projection.visible_cell_ids, ())
        self.assertTrue(projection.chunks)
        self.assertTrue(set(projection.visible_chunk_ids).issubset(set(query.chunk_ids)))
        for chunk in projection.chunks:
            self.assertGreaterEqual(len(chunk.points), 6)
            self.assertEqual(len(chunk.points) % 2, 0)

    def test_cache_round_trip_and_corruption_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            info = write_chunk_visibility_cache(self.index, temporary)
            loaded = load_chunk_visibility_cache(info.directory)
            self.assertEqual(loaded.stable_hash, self.index.stable_hash)
            self.assertEqual(loaded.chunks, self.index.chunks)
            self.assertEqual(loaded.nodes, self.index.nodes)
            report = verify_chunk_visibility_cache(info.directory)
            self.assertTrue(report["valid"])

            manifest_path = info.directory / "visibility.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index_path = info.directory / manifest["files"]["index"]["path"]
            payload = bytearray(index_path.read_bytes())
            payload[-1] ^= 0xFF
            index_path.write_bytes(payload)
            with self.assertRaises(ChunkVisibilityError):
                load_chunk_visibility_cache(info.directory)

    def test_frequency_128_query_prunes_work(self) -> None:
        topology = generate_dual_topology(128)
        layout = build_chunk_layout(topology)
        index = build_chunk_visibility_index(topology, layout)
        viewport = SphereViewport(zoom=4.0)
        query = index.query(
            layout, viewport.yaw, viewport.pitch, viewport.zoom, 900, 700
        )
        self.assertLess(query.candidate_cells, topology.cell_count // 2)
        self.assertLess(query.tested_chunks, layout.chunk_count // 2)
        self.assertLess(query.visited_nodes, index.node_count)


if __name__ == "__main__":
    unittest.main()
