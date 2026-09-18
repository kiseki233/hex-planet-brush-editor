# Production LOD and Multi-Level Aggregate Cache

## LOD policy

The production controller estimates visible cells from planet size, viewport aspect, and zoom, then applies the same hysteresis boundaries as the detailed design:

```text
LOD0 out > 17×17, return < 15×15
LOD1 out > 33×33, return < 31×31
LOD2 out > 65×65, return < 63×63
LOD3 out > 129×129, return < 127×127
LOD4 aggregate far view
```

## Hierarchy

Chunks are grouped spatially inside each of the 20 icosahedron base faces. Binary parents store a spherical cap and refer to two children. A view chooses the highest nodes whose projected diameter is below the target pixel size.

## Cache

```text
art/maps/<map>/lod/aggregate_v1/
├─ aggregate.json
└─ nodes/
   ├─ node_00000.json
   ├─ node_00000.png
   └─ ...
```

A selected node may be summarized directly from its descendant Pack chunks. This avoids writing every intermediate node before the first far view. Existing valid nodes are reused.

A saved chunk edit invalidates its leaf and every parent up to the base-face root. Unrelated branches remain valid. The cache is derived and may be deleted safely.

## Editing rule

LOD4 does not expose per-cell picking. Detail instances are restored automatically after zooming into LOD3 or closer.
