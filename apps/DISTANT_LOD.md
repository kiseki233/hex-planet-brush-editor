# Distant and Aggregate LOD Cache Formats v1.3.1

All files under a map's `lod/` directory are derived. Deleting them never deletes map authority.

## Per-chunk and planet overview cache

```text
art/maps/<map>/lod/distant_v1/
├─ chunks/
├─ planet_512x256.png
└─ planet.json
```

Chunk signatures include topology/layout identity, uint16 states, brush UIDs, content hashes, and missing-resource state. The planet overview is built by streaming Pack chunks.

## Multi-level aggregate cache

```text
art/maps/<map>/lod/aggregate_v1/
├─ aggregate.json
└─ nodes/
```

The aggregate hierarchy sits above individual chunks. The renderer selects coarse or fine nodes by projected pixel diameter. Selected nodes can be generated on demand; an optional full build creates all levels.

## Invalidation

- Painting or erasing changes one authoritative chunk.
- Saving invalidates the affected chunk cache, planet signature, aggregate leaf, and all aggregate ancestors.
- Replacing a brush at the same path preserves UID but changes content hash and therefore cache identity.
- Unaffected branches remain reusable.

## Production software-globe surface (v1.0.3)

The Tk main globe uses an additional derived cache:

```text
art/maps/<map-name>/lod/software_surface_v1/
├─ planet_1024x512.png
└─ surface.json
```

The image is generated from the authoritative production Pack index and brush content hashes.
It stores a chunk-scale equirectangular representation and is sampled onto the sphere according
to the current yaw and pitch. It is not authoritative map data and can be deleted safely.
