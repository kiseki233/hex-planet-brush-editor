# Frequency-1004 Production GPU Editor v1.3.1

## Goal

Edit the complete 10,080,162-cell planet without materializing the complete topology or allocating 10,080,150 GPU instances.

## Compact indexes

```text
art/maps/.topology/ico_dual_f1004_v2/
├─ production.json
├─ production_chunks.idx
├─ production_visibility.json
└─ production_visibility.idx
```

The layout describes 39,060 procedural chunks. The visibility index contains one conservative spherical cap per chunk and 8,191 deterministic hierarchy nodes.

## View updates

- The camera traverses only intersecting visibility nodes.
- Detailed LODs request only candidate chunks.
- A bounded scheduler incrementally transitions between views.
- Entering Pack blocks are decompressed and CRC-checked without long-lived session caching.
- LOD4 selects multi-level aggregate nodes instead of cells.
- Selected aggregate node caches are generated on demand; a full hierarchy prebuild is optional.

## Edit behavior

At LOD0–LOD3, native picking resolves the nearest visible normal hexagon. The authoritative owner allocates/reuses a local brush ID, writes the uint16 state, marks one dirty chunk, and returns one GPU patch. LOD4 is view-only; zooming in restores editing.

## Save behavior

Dirty chunks are appended to Pack storage and `index.bin` is atomically replaced. Saved edits invalidate only the affected aggregate ancestor chains. Pack compaction remains transactional and recoverable.

## Current external boundary

The complete software path is implemented. Real Win32/WGL behavior, DPI, driver compatibility, and final visual seam appearance must be run on the target Windows GPU using the included diagnostics.
