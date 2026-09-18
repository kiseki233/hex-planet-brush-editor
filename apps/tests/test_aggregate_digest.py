from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.aggregate_lod import AggregateLodCache, ProductionAggregateHierarchy
from app.production_layout import ProductionChunkLayout
from app.production_topology import ProductionTopology
from app.production_visibility import build_production_visibility_index
from app.sphere_map_store import SphereMapStore


def reference_signature(prefix, node_id, chunk_ids, session, store, records_by_uid, color_cache):
    """The original per-cell hashing, byte for byte."""
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(node_id.to_bytes(4, "little"))
    for chunk_id in chunk_ids:
        values = store.read_chunk_values(session, chunk_id)
        digest.update(chunk_id.to_bytes(4, "little"))
        for value in values:
            digest.update(int(value).to_bytes(2, "little"))
            local_id = value & 0x0FFF
            if local_id != 0:
                entry = session.brush_entries.get(local_id)
                record = None if entry is None else records_by_uid.get(entry.brush_uid)
                if record is not None:
                    digest.update(record.uid.encode("utf-8"))
                    digest.update(record.content_hash.encode("ascii"))
    return digest.hexdigest()


class AggregateDigestCompatibilityTest(unittest.TestCase):
    """Bulk hashing must not change any node signature.

    A changed signature would silently orphan every already-built node in a
    user's cache and force a full multi-minute rebuild.
    """

    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)
        cls.visibility = build_production_visibility_index(cls.layout)
        cls.hierarchy = ProductionAggregateHierarchy(cls.visibility, cls.layout)

    def _fixture(self, temporary, painted=False):
        store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=16)
        session = store.create_blank("planet", self.layout)
        records = {}
        if painted:
            record = _FakeRecord("uid-a", "地形/a.png", "abc123")
            records[record.uid] = record
            cells = [c for c in self.layout.chunk_cell_ids(0) if c >= 12][:40]
            session.set_cells(store, {c: (record.uid, record.relative_path, 2) for c in cells})
            store.save(session)
        return store, session, records

    def _check(self, painted):
        with tempfile.TemporaryDirectory() as temporary:
            store, session, records = self._fixture(temporary, painted=painted)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            checked = 0
            for node_id in range(min(6, self.hierarchy.node_count)):
                chunk_ids = self.hierarchy.descendant_chunks(node_id)
                if not chunk_ids:
                    continue
                summary = cache._build_direct(
                    node_id, chunk_ids, self.layout, session, store, records
                )
                expected = reference_signature(
                    b"aggregate-direct-v1", node_id, chunk_ids,
                    session, store, records, cache.color_cache,
                )
                self.assertEqual(summary.signature, expected, f"node {node_id}")
                checked += 1
            self.assertGreater(checked, 0)

    def test_blank_map_signatures_are_unchanged(self):
        self._check(painted=False)

    def test_painted_map_signatures_are_unchanged(self):
        self._check(painted=True)

    def test_leaf_signatures_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, session, records = self._fixture(temporary, painted=True)
            cache = AggregateLodCache(Path(temporary) / "brushes")
            checked = 0
            for node_id in range(self.hierarchy.node_count):
                if not self.hierarchy.nodes[node_id].is_leaf:
                    continue
                chunk_ids = self.hierarchy.descendant_chunks(node_id)
                if not chunk_ids:
                    continue
                summary = cache._build_leaf(
                    node_id, chunk_ids, self.layout, session, store, records
                )
                expected = reference_signature(
                    b"aggregate-leaf-v1", node_id, chunk_ids,
                    session, store, records, cache.color_cache,
                )
                self.assertEqual(summary.signature, expected, f"leaf {node_id}")
                checked += 1
                if checked >= 5:
                    break
            self.assertGreater(checked, 0)


class BatchChunkReadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = ProductionTopology(16)
        cls.layout = ProductionChunkLayout(cls.topology, tile_side=4)

    def test_batch_read_matches_single_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("planet", self.layout)
            cells = [c for c in self.layout.chunk_cell_ids(2) if c >= 12][:5]
            session.set_cells(store, {c: ("uid-a", "a/a.png", 1) for c in cells})
            store.save(session)

            chunk_ids = tuple(range(min(10, self.layout.chunk_count)))
            batch = store.read_chunk_values_many(session, chunk_ids)
            self.assertEqual(set(batch), set(chunk_ids))
            for chunk_id in chunk_ids:
                self.assertEqual(batch[chunk_id], store.read_chunk_values(session, chunk_id))

    def test_batch_read_rejects_out_of_range_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("planet", self.layout)
            with self.assertRaises(Exception):
                store.read_chunk_values_many(session, (self.layout.chunk_count + 5,))

    def test_batch_read_does_not_populate_the_session_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SphereMapStore(Path(temporary) / "maps", chunks_per_pack=4)
            session = store.create_blank("planet", self.layout)
            session.loaded_chunks.clear()
            store.read_chunk_values_many(session, (0, 1, 2))
            self.assertEqual(session.loaded_chunks, {})


class _FakeRecord:
    """Enough of BrushRecord for the aggregate summariser."""

    state = "active"
    average_rgb = (120, 90, 60)

    def __init__(self, uid, relative_path, content_hash):
        self.uid = uid
        self.relative_path = relative_path
        self.content_hash = content_hash
        self.category_path = relative_path.rsplit("/", 1)[0]


if __name__ == "__main__":
    unittest.main()
