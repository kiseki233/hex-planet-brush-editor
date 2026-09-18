# Brush LOD and Asynchronous Loading Format

## Derived cache identity

```text
art/brushes/.catalog/cache/lod_v1/<brushUid>/<contentHash>/
```

The content hash prevents stale cache reuse after a source image changes. A same-path source replacement keeps the stable UID but receives a new hash directory.

## LOD files

```text
lod0_520.png  effective 512×512 + 4 px padding
lod1_264.png  effective 256×256 + 4 px padding
lod2_136.png  effective 128×128 + 4 px padding
lod3_72.png   effective  64×64  + 4 px padding
manifest.json
```

Padding pixels copy the nearest valid edge pixel, including corners. Files are 8-bit RGB or RGBA PNG and use nearest-neighbor source reduction.

## Cache authority

The source image under `art/brushes/` is authoritative. The entire `lod_v1` tree may be deleted and regenerated. Maps continue to store brush UID references rather than cache paths.

## Pack worker contract

`SphereMapStore.read_chunk_values()` reads and validates one chunk without mutating `session.loaded_chunks`. `AsyncViewportChunkLoader` runs that method in worker threads and applies completed values on the Tk thread.

The queue is bounded, stale data is discarded, dirty data is never evicted, and failed reads are held until the chunk leaves and re-enters the requested viewport.
