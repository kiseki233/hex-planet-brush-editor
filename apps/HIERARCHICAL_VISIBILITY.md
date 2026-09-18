# Hierarchical Chunk Visibility Format v1

## Purpose

The visibility index removes the previous requirement to project every cell before deciding which chunks may be visible.

The authoritative topology and map data do not change. This index is a rebuildable topology cache derived from:

- `topology.stable_hash`
- `chunk_layout.stable_hash`
- deterministic chunk membership
- deterministic dual-corner boundary extraction

## Cache files

```text
art/maps/.topology/ico_dual_f<frequency>_v1/
├─ visibility.json
└─ visibility.idx
```

Deleting these files does not delete map content. They can be regenerated from the topology and chunk layout.

## Per-chunk data

Each chunk stores:

- stable `ChunkId`
- unit-vector spherical-cap center
- angular cap radius
- sorted dual-triangle IDs forming the chunk boundary support set

The cap includes all cell centers and all incident dual corners belonging to the chunk. It is conservative: a visible chunk may be retained as a false positive, but a truly visible chunk must not be rejected.

Boundary support points are used by the distant renderer. It projects only these chunk-boundary dual corners and builds a screen-space convex hull. It does not project every cell in distant mode.

## Hierarchy

The chunk caps are arranged in a deterministic binary hierarchy.

- Default leaf size: 8 chunks.
- Split axis: widest chunk-center coordinate range.
- Split order: coordinate value, then `ChunkId`.
- Split point: stable median.
- Internal nodes store a conservative spherical cap for all descendant chunks.
- Leaves store sorted `ChunkId` values.

The hierarchy hash includes all chunk caps, boundary triangle IDs, nodes, leaf membership, topology identity, and layout identity.

## Query stages

For each viewport update:

1. Rotate a hierarchy node cap into camera space.
2. Reject caps completely behind the visible hemisphere.
3. Reject caps whose conservative screen bound is outside the canvas.
4. Descend only into retained nodes.
5. Test retained leaf chunks with their own caps.
6. Return sorted candidate `ChunkId` values.

Detailed mode then projects only cells belonging to candidate chunks. Distant mode projects only candidate chunk boundaries.

## Safety and validation

The reader verifies:

- magic and format version
- record sizes and total file length
- CRC32 and SHA-256
- topology and layout hashes
- stable hierarchy hash
- contiguous chunk and node IDs
- valid leaf membership
- exact one-time coverage of every chunk
- valid boundary triangle references when topology is supplied

Corruption is reported instead of silently rebuilding over the damaged cache.

## Current scale boundary

The v0.7.0 Python/Tk implementation exposes frequency 128 for stress testing. The hierarchical query removes full-cell scanning from distant rendering and reduces detailed projection to candidate chunks, but it is not the final frequency-1004 renderer.

A complete production implementation still needs native or streaming topology generation and GPU rendering.
