# Chunk and Pack Storage Format v1

## 1. Stable chunk layout

The chunk layout belongs to a specific topology hash and is stored under the shared topology directory.

```text
art/maps/.topology/ico_dual_f<frequency>_v1/
├─ chunks.json
└─ chunks.idx
```

Each cell is assigned to the nearest of the 20 base-face centers. Exact ties use the lowest base-face ID. Inside each face, chunks first grow through sorted reciprocal neighbor links with deterministic breadth-first traversal. Undersized adjacent chunks are then merged or split again with connectivity checks so tiny remainder chunks are not retained.

The result provides:

- `CellId -> ChunkId`
- `CellId -> local index inside chunk`
- `ChunkId -> base face and ordered CellId list`

Every chunk is connected and contains at most the configured target count.

## 2. `chunks.idx`

Header fields:

- magic
- format version
- frequency
- cell count
- chunk count
- target cells
- chunk-record size
- topology SHA-256

Sections after the header:

1. `uint32[cellCount]` CellId-to-ChunkId table.
2. `uint16[cellCount]` CellId-to-local-index table.
3. Fixed chunk records containing base face, member offset, and member count.
4. Flattened `uint32[cellCount]` CellId member array.

`chunks.json` stores the CRC32, SHA-256, topology hash, layout hash, counts, and partition method.

## 3. Sphere map `index.bin`

`index.bin` contains one fixed record per chunk and is replaced atomically after changed blocks are safely appended.

Each record contains:

- `chunkId`
- `packId`
- `offset`
- `compressedSize`
- `rawSize`
- `compressionType`
- `crc32`
- `cellCount`
- `flags`

The index header includes both topology and chunk-layout SHA-256 identities.

## 4. Pack files

Each pack starts with a header that identifies:

- pack format version
- pack ID
- configured chunks per pack
- topology hash
- chunk-layout hash

Each chunk block has its own header and payload. The raw payload is a little-endian sequence of `uint16` cell states in the chunk layout's fixed CellId order.

Compression types:

```text
0 = raw
1 = zlib
```

Zlib is used only when the compressed payload is smaller than the raw payload.

## 5. Incremental save order

1. Write an updated brush table when required.
2. Append every dirty chunk block to its assigned pack.
3. Flush each changed pack to disk.
4. Build a complete temporary `index.bin` with the new offsets.
5. Flush the temporary index.
6. Atomically replace the previous `index.bin`.

If the program stops before the index replacement, the old index still points to the previous valid blocks. Newly appended but unreferenced blocks are harmless and can be reclaimed by a future compaction tool.

## 6. Error isolation

Opening a map validates the manifest and index structure but does not decompress all chunks. A damaged block is reported when that chunk is requested or during explicit full verification.

This allows other valid chunks to remain readable when one chunk is damaged.


## v0.8.0 Pack compaction

Incremental save remains append-only. Pack compaction copies only the blocks referenced by the current `index.bin` into a staged Pack set, verifies the staged set, switches `data/` and `index.bin` with recovery backups, verifies the installed set, and then removes the backups. See `PACK_COMPACTION.md`.
