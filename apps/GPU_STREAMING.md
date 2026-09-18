# Visible GPU Instance Streaming v1.3.1

`VisibleGpuInstanceStream` and `ProductionGpuStreamingController` are the production-scale data path. They never allocate one instance for every editable cell on the planet.

## View path

```text
GpuViewRequest
→ production visibility hierarchy
→ candidate ChunkIds
→ frame-budget scheduler
→ direct Pack reads and validation
→ dense active instance stream
→ GpuStreamPatch / GpuBatchResetPatch
```

The scheduler prioritizes chunks nearest the visible screen center and limits additions/removals per update. Stale camera requests are replaced by the latest request.

## Dense instances

- Entering chunks append ordinary hexagon instances.
- Hidden pentagons are skipped.
- Leaving chunks use swap-remove.
- Moved instances receive corrected slots.
- The active VBO is proportional to the current view, not 10,080,150 hexagons.

## Automatic LOD

LOD0–LOD3 reset the active stream to the matching brush texture size. LOD4 removes detail instances and replaces them with selected multi-level aggregate nodes. Hysteresis prevents repeated switching around a boundary.

## Texture residency

Layer identity is:

```text
brushUid + contentHash + lodLevel
```

The empty and missing layers are pinned. Brush layers have visible-instance reference counts. After a short grace period at zero references, the layer is released and its numeric slot becomes reusable. Reuse is sent to the native renderer as a replacement upload, so array capacity does not grow forever during world traversal.

## Memory behavior

- Stream Pack reads do not populate `session.loaded_chunks`.
- Clean inactive chunks are released.
- Dirty chunks remain authoritative until save succeeds.
- Texture cache files are derived data.
- GPU instance and texture residency follow the visible working set.

## Platform boundary

The stream logic, dense deltas, texture replacement contract, production queries, Pack integration, and save/reopen path are automated and passing. Actual WGL driver execution must be checked on the target Windows PC with the supplied diagnostic.
