# Spherical Editor Behavior v1.3.1

## Correctness editor

The existing Tk correctness editor remains available for materialized frequencies through 128.

Its detailed pipeline is:

1. Query the deterministic test chunk hierarchy.
2. Obtain conservative candidate chunks without scanning every cell.
3. Select LOD with hysteresis.
4. In LOD0–LOD3, project cells inside candidate chunks.
5. In LOD4, project chunk-boundary dual corners.
6. Request detailed Pack chunks or distant thumbnails only when needed.

It supports exact projected-polygon hit testing and CPU/Tk painting.

## Production editor

`打开千万格生产 GPU 编辑器` uses frequency 1004 and the procedural production path.

Preparation:

```text
ProductionTopology
→ ProductionChunkLayout
→ ProductionVisibilityIndex
→ complete Pack map
```

Interactive camera pipeline:

```text
yaw / pitch / zoom / viewport
→ compact hierarchy query
→ candidate ChunkIds
→ worker Pack reads
→ VisibleGpuInstanceStream
→ native GPU stream patch
```

The production Tk canvas is a compact chunk-level diagnostic preview. On Windows, the native GPU window is the detailed editable surface.

## Authoritative editing

- The native window submits a CellId, tool, brush UID, and rotation.
- Tk validates that the CellId is an editable hexagon.
- `SphereMapSession.set_cell()` updates the uint16 state and dirty chunk.
- The GPU receives one cell patch and optional texture upload.
- Save uses the normal append-and-atomic-index Pack path.
- Reopening the map restores the same brush UID and rotation.

## Visibility and memory boundary

The production hierarchy contains chunk-level bounds only. It does not store ten million cell visibility records.

Only queried chunks are expanded into CellIds and instance geometry. Stream reads do not populate the long-lived session cache, and inactive clean chunks are released.

## Distant boundary

The production editor currently streams detailed chunk instances for a restricted zoom range. A separate higher-than-chunk aggregated far-distance geometry hierarchy for displaying the complete globe with very few regions is still not implemented.

## Platform boundary

The production editor's Tk preparation and data path run cross-platform. The detailed native GPU surface requires Windows and an OpenGL 3.3-capable driver. This package has not been executed on a real WGL driver in the Linux build environment.
