# Procedural Production Topology v2

## Purpose

The materialized v1 topology stores every center, neighbor list, incident triangle list, and triangle center. That representation is useful for test frequencies but is not suitable for 10,080,162 cells in Python.

v2 keeps the same global CellId ordering while calculating geometry on demand.

## Stable CellId regions

```text
0–11                          original icosahedron vertices
12–face_start-1               ordered interior points on 30 base edges
face_start–cell_count-1       face interior barycentric points
```

The reverse mapping reconstructs one or more `(baseFace, weightA, weightB, weightC)` addresses for any CellId.

## On-demand operations

- `cell_center(CellId)` normalizes the barycentric weighted base vertices.
- `cell_neighbors(CellId)` applies six barycentric lattice directions across every face representation and de-duplicates shared-edge results.
- `cell_corners(CellId)` forms dual corners from adjacent ordered neighbors.
- `owner_address(CellId)` assigns shared edge and vertex points to the lowest containing base-face id.

The procedural results are tested against the complete v1 materialized topology for small frequencies.

## Production chunks

Face coordinates are divided into 16×16 barycentric tiles. Small boundary tiles are deterministically merged with a connected cardinal neighbor. The cache stores tile-key groups and cell counts, not ten million member IDs.

A requested chunk enumerates at most a small group of local tiles and produces its CellIds on demand.

## Production visibility hierarchy

v1.0.3 uses a second compact cache beside the chunk layout.

Each production chunk stores a conservative unit-sphere cap and one seed CellId. A deterministic binary hierarchy groups the caps. The frequency-1004 index contains 39,060 chunk bounds and 8,191 nodes.

A viewport query traverses only intersecting nodes and returns candidate ChunkIds plus a candidate-cell estimate. It does not enumerate all 10,080,162 cells.

Cache files:

```text
production_visibility.json
production_visibility.idx
```

Both files are tied to the production topology hash and layout hash and are protected by CRC32, SHA-256, and a stable hierarchy hash.
