from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from app.brush_catalog import BrushCatalog
from app.chunk_layout import build_chunk_layout
from app.gpu_batch import (
    INSTANCE_STRUCT,
    MESH_VERTEX_STRUCT,
    GpuRenderBatchBuilder,
    build_texture_array,
    rotate_uv,
    unit_hex_mesh_bytes,
)
from app.gpu_native import FRAGMENT_SHADER_SOURCE, VERTEX_SHADER_SOURCE, native_gpu_supported
from app.sphere_map_store import SphereMapStore
from app.topology import generate_dual_topology
from tests.test_helpers import write_rgb_png


class GpuBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = generate_dual_topology(4)
        cls.layout = build_chunk_layout(cls.topology, target_cells=32)

    def _fixture(self, temporary: str):
        root = Path(temporary)
        brush_root = root / "brushes"
        map_root = root / "maps"
        source = brush_root / "地形" / "green.png"
        write_rgb_png(source, rgb=(24, 160, 72))
        catalog = BrushCatalog(brush_root)
        record = catalog.scan().active_records[0]
        records = {record.uid: record}
        store = SphereMapStore(map_root)
        session = store.create_blank("sphere", self.layout)
        editable = [cell_id for cell_id in range(self.topology.cell_count) if cell_id >= 12]
        session.set_cell(editable[0], store, record.uid, record.relative_path, 1)
        session.set_cell(editable[1], store, record.uid, record.relative_path, 5)
        store.save(session)
        return brush_root, store, session, records, editable

    def test_repeated_brush_uses_one_texture_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, store, session, records, editable = self._fixture(temporary)
            builder = GpuRenderBatchBuilder(brush_root)
            batch = builder.build(
                self.topology,
                self.layout,
                session,
                store,
                records,
                editable[:12],
                2,
            )
            self.assertEqual(batch.instance_count, 12)
            self.assertEqual(batch.texture_layer_count, 3)
            brush_layers = [layer for layer in batch.texture_layers if layer.kind == "brush"]
            self.assertEqual(len(brush_layers), 1)
            painted = [instance for instance in batch.instances if instance.texture_layer == brush_layers[0].layer]
            self.assertEqual({instance.rotation for instance in painted}, {1, 5})
            self.assertEqual(len(batch.instance_bytes()), batch.instance_count * INSTANCE_STRUCT.size)

    def test_texture_array_is_rgba_and_matches_lod_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, store, session, records, editable = self._fixture(temporary)
            batch = GpuRenderBatchBuilder(brush_root).build(
                self.topology,
                self.layout,
                session,
                store,
                records,
                editable[:4],
                3,
            )
            textures = build_texture_array(batch)
            self.assertEqual((textures.width, textures.height), (72, 72))
            self.assertEqual(textures.layer_count, 3)
            self.assertEqual(textures.byte_size, 72 * 72 * 4 * 3)
            self.assertEqual(textures.layer_keys[0], "__empty__")
            self.assertEqual(textures.layer_keys[1], "__missing__")

    def test_instance_tangent_basis_is_orthonormal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, store, session, records, editable = self._fixture(temporary)
            batch = GpuRenderBatchBuilder(brush_root).build(
                self.topology,
                self.layout,
                session,
                store,
                records,
                editable[:20],
                1,
            )
            for instance in batch.instances:
                center = instance.center
                u = instance.tangent_u
                v = instance.tangent_v
                dot_cu = sum(a * b for a, b in zip(center, u))
                dot_cv = sum(a * b for a, b in zip(center, v))
                dot_uv = sum(a * b for a, b in zip(u, v))
                self.assertAlmostEqual(dot_cu, 0.0, places=6)
                self.assertAlmostEqual(dot_cv, 0.0, places=6)
                self.assertAlmostEqual(dot_uv, 0.0, places=6)
                self.assertAlmostEqual(math.sqrt(sum(value * value for value in u)), 1.0, places=6)
                self.assertAlmostEqual(math.sqrt(sum(value * value for value in v)), 1.0, places=6)
                self.assertGreater(instance.radius, 0.0)

    def test_batch_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, store, session, records, editable = self._fixture(temporary)
            builder = GpuRenderBatchBuilder(brush_root)
            first = builder.build(
                self.topology,
                self.layout,
                session,
                store,
                records,
                editable[:24],
                2,
            )
            second = builder.build(
                self.topology,
                self.layout,
                session,
                store,
                records,
                editable[:24],
                2,
            )
            self.assertEqual(first.stable_hash, second.stable_hash)
            self.assertEqual(first.instance_bytes(), second.instance_bytes())

    def test_missing_brush_uses_shared_missing_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("sphere", self.layout)
            editable = [cell_id for cell_id in range(self.topology.cell_count) if cell_id >= 12]
            session.set_cell(editable[0], store, "missing-uid", "missing.png", 3)
            session.set_cell(editable[1], store, "missing-uid", "missing.png", 4)
            store.save(session)
            batch = GpuRenderBatchBuilder(root / "brushes").build(
                self.topology, self.layout, session, store, {}, editable[:2], 1
            )
            self.assertEqual(batch.texture_layer_count, 2)
            self.assertEqual({instance.texture_layer for instance in batch.instances}, {1})

    def test_background_batch_build_does_not_populate_session_chunk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            brush_root, store, session, records, editable = self._fixture(temporary)
            session.loaded_chunks.clear()
            GpuRenderBatchBuilder(brush_root).build(
                self.topology, self.layout, session, store, records, editable[:12], 2
            )
            self.assertEqual(session.loaded_chunks, {})

    def test_unit_hex_mesh_has_six_triangles(self) -> None:
        mesh = unit_hex_mesh_bytes()
        self.assertEqual(len(mesh), 18 * MESH_VERTEX_STRUCT.size)

    def test_six_uv_rotations_return_after_full_turn(self) -> None:
        original = (0.78, 0.31)
        current = original
        for _ in range(6):
            relative = (current[0] - 0.5, current[1] - 0.5)
            current = rotate_uv((0.5 + relative[0], 0.5 + relative[1]), 1)
        self.assertAlmostEqual(current[0], original[0], places=6)
        self.assertAlmostEqual(current[1], original[1], places=6)
        self.assertNotEqual(rotate_uv(original, 1), original)

    def test_shader_contract_uses_instancing_texture_array_and_shader_rotation(self) -> None:
        self.assertIn("sampler2DArray", FRAGMENT_SHADER_SOURCE)
        self.assertIn("inRotation * 60.0", VERTEX_SHADER_SOURCE)
        self.assertIn("inLayer", VERTEX_SHADER_SOURCE)
        self.assertIn("uEffectiveRatio", VERTEX_SHADER_SOURCE)

    def test_native_backend_reports_platform_support_without_import_failure(self) -> None:
        self.assertEqual(native_gpu_supported(), __import__("os").name == "nt")


if __name__ == "__main__":
    unittest.main()
