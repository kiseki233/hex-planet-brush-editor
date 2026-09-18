# Topology cache format v1

## Directory

```text
art/maps/.topology/ico_dual_f<frequency>_v1/
├─ topology.json
├─ cells.bin
└─ triangles.bin
```

The cache is derived data. It can be regenerated from the fixed topology type, frequency, generator version, and orientation.

## Stable CellId order

CellIds are assigned in this fixed order:

1. The 12 oriented base icosahedron vertices.
2. Interior points on the 30 lexicographically sorted base edges.
3. Interior points of each of the 20 oriented base faces in fixed barycentric order.

This gives exactly:

```text
10 × frequency² + 2
```

The first 12 CellIds are always the pentagons.

## `cells.bin`

Header layout:

```text
8 bytes  magic
uint32   binary version
uint32   frequency
uint32   cell count
uint32   triangle count
```

Each 64-byte cell record contains:

```text
float32 × 3   unit-sphere center
uint8         degree, 5 or 6
uint8         flags; bit 0 means pentagon
uint16        reserved
uint32 × 6    clockwise neighbor CellIds
uint32 × 6    clockwise incident triangle IDs used as dual corners
```

Unused sixth entries for a pentagon are `0xffffffff`.

## `triangles.bin`

The header contains the same version, frequency, cell count, and triangle count fields.

Each 12-byte record stores one normalized primal-triangle center as three float32 values. These centers are the dual-cell polygon corners referenced by `cells.bin`.

## `topology.json`

The manifest records:

- Topology type, frequency, generator version, and orientation.
- Cell, triangle, edge, pentagon, and hexagon counts.
- Stable topology hash.
- Binary record sizes.
- CRC32 and SHA-256 values for both binary files.

## Current limitation

The v1 cache stores complete in-memory topology results and is used for correctness validation. It is not yet split into the final approximately-256-cell chunks and does not yet include viewport lookup acceleration.


The deterministic chunk-layout format built on top of this cache is documented in `CHUNK_STORAGE_FORMAT.md`.
