from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.gpu_edit import GpuBatchResetPatch, GpuEditBridge, GpuViewRequest
from app.gpu_stream import VisibleGpuInstanceStream, apply_instance_delta
from app.production_layout import ProductionChunkLayout
from app.production_streaming import ProductionGpuStreamingController
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore


class ProductionStreamingTests(unittest.TestCase):
    def test_bridge_routes_stream_view_requests(self) -> None:
        bridge = GpuEditBridge()
        request_id = bridge.submit_view(-0.2, 0.3, 9.0, 1200, 800)
        requests = bridge.poll_requests()
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertIsInstance(request, GpuViewRequest)
        self.assertEqual(request.request_id, request_id)
        self.assertEqual((request.width, request.height), (1200, 800))

    def test_dense_delta_matches_visible_stream_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(64)
            layout = ProductionChunkLayout(topology)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            stream = VisibleGpuInstanceStream(root / "brushes", lod_level=3)
            first = stream.update_visible_chunks(
                topology, layout, session, store, {}, [0, 1, 2, 3]
            )
            mirror = [item.instance for item in first.added]
            mapping = {instance.cell_id: index for index, instance in enumerate(mirror)}
            second = stream.update_visible_chunks(
                topology, layout, session, store, {}, [2, 3, 4, 5]
            )
            self.assertEqual(second.changed, ())
            apply_instance_delta(
                mirror,
                mapping,
                second.removed_cell_ids,
                second.changed,
                second.added,
            )
            self.assertEqual(mirror, stream.instances)
            self.assertEqual(mapping, stream.cell_to_slot)

    def test_controller_keeps_only_visible_production_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(128)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            controller = ProductionGpuStreamingController(
                topology,
                layout,
                visibility,
                session,
                store,
                {},
                root / "brushes",
                lod_level=3,
            )
            first = controller.update_view(-0.35, 0.25, 12.0, 1100, 760)
            batch = controller.initial_batch()
            self.assertEqual(batch.instance_count, first.update.instance_count)
            self.assertEqual(session.loaded_chunks, {})

            limit = controller.apply_hardware_texture_layer_limit(512)
            self.assertEqual(limit, 512)
            self.assertEqual(controller.stream.maximum_texture_layers, 512)
            self.assertLess(first.update.instance_count, topology.cell_count // 10)
            second = controller.update_view(1.7, -0.4, 12.0, 1100, 760)
            patch = controller.patch_for_frame(7, second)
            self.assertEqual(patch.instance_count, controller.stream.instance_count)
            self.assertTrue(patch.added or patch.removed_cell_ids)
            self.assertEqual(session.loaded_chunks, {})

    def test_automatic_lod_switches_between_aggregate_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(128)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            controller = ProductionGpuStreamingController(
                topology, layout, visibility, session, store, {}, root / "brushes",
                lod_level=4, automatic_lod=True,
            )
            far = controller.update_view(0.0, 0.0, 1.0, 1100, 760)
            self.assertEqual(far.lod.level, 4)
            self.assertIsNotNone(far.aggregate_selection)
            # The far view shows the saved-surface sphere: no proxy mosaic
            # instances, and the brush stays live at every zoom.
            self.assertEqual(far.reset_batch.instance_count, 0)
            far_patch = controller.patch_for_frame(1, far)
            self.assertTrue(far_patch.editable)
            near = controller.update_view(0.0, 0.0, 30.0, 1100, 760)
            self.assertLess(near.lod.level, 4)
            self.assertGreater(near.update.instance_count, 0)
            near_patch = controller.patch_for_frame(2, near)
            # Aggregate mode leaves the producer's internal stream at LOD3 but
            # the consumer owns a different (surface-only) instance batch. The
            # first detailed frame must therefore be a full reset, even when the
            # selected detail level is also LOD3.
            self.assertIsInstance(near_patch, GpuBatchResetPatch)
            self.assertEqual(
                near_patch.batch.instance_count,
                near.update.instance_count,
            )
            self.assertTrue(near_patch.editable)

    def test_budgeted_controller_requests_follow_up_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(64)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            controller = ProductionGpuStreamingController(
                topology, layout, visibility, session, store, {}, root / "brushes",
                lod_level=3, stream_add_budget=2, stream_remove_budget=3,
            )
            frame = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            self.assertTrue(frame.has_more)
            percentages = [frame.load_percent]
            self.assertLess(frame.loaded_chunk_count, frame.total_chunk_count)
            for _ in range(100):
                frame = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
                percentages.append(frame.load_percent)
                if not frame.has_more:
                    break
            self.assertFalse(frame.has_more)
            self.assertEqual(frame.loaded_chunk_count, frame.total_chunk_count)
            self.assertEqual(frame.load_percent, 100)
            self.assertEqual(percentages, sorted(percentages))
            patch = controller.patch_for_frame(9, frame)
            self.assertEqual(patch.loaded_chunk_count, frame.loaded_chunk_count)
            self.assertEqual(patch.total_chunk_count, frame.total_chunk_count)
            self.assertEqual(set(frame.update.active_chunk_ids), set(frame.query.chunk_ids))
            controller.close()

    def test_consumer_reset_keeps_producer_progress_and_emits_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = ProductionTopology(64)
            layout = ProductionChunkLayout(topology)
            visibility = build_production_visibility_index(layout)
            store = SphereMapStore(root / "maps")
            session = store.create_blank("production", layout)
            controller = ProductionGpuStreamingController(
                topology,
                layout,
                visibility,
                session,
                store,
                {},
                root / "brushes",
                lod_level=3,
                stream_add_budget=2,
                stream_remove_budget=3,
            )
            first = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            self.assertTrue(first.has_more)
            controller.request_consumer_reset()
            second = controller.update_view(0.0, 0.0, 8.0, 1100, 760)
            self.assertGreater(
                second.update.instance_count,
                first.update.instance_count,
            )
            patch = controller.patch_for_frame(9, second)
            self.assertIsInstance(patch, GpuBatchResetPatch)
            controller.close()


if __name__ == "__main__":
    unittest.main()
