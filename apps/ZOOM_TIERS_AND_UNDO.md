# Six-tier zoom ladder and per-stroke undo

## What changed

Painting used to require `zoom >= 36` in the sphere viewport and 9 px/cell in the
2D net editor. Both gates existed because the click was resolved by testing the
mouse position against the rendered detail polygons, so no polygons meant no
edit. The zoom range is 0.8–512, which left roughly 94 percent of the
logarithmic range unpaintable, and the one paintable band showed a screen radius
of about 27 cells while the brush could be 250 cells in radius.

Picking is now analytic and independent of what is rendered, so every zoom level
is editable. What changes between levels is how coarse the visual feedback is.

## The ladder

`app/zoom_tiers.py` defines six tiers. Each floor is about 2.9x apart in zoom,
which is about 8.5x apart in visible cells, so the ladder is uniform on a
logarithmic scale.

```text
tier        zoom        visible cells    px/cell   render               feedback
L0 原图     >= 110      <= 290           >= 57     512x512 per cell     1 cell
L1 精细     38 - 110    290 - 2.4k       20 - 57   256x256 per cell     1 cell
L2 常用     13 - 38     2.4k - 20k       6.7 - 20  128x128 per cell     1 cell
L3 区域     4.5 - 13    20k - 170k       2.3 - 6.7 64x64 per cell       1 cell
L4 地块     1.6 - 4.5   170k - 1.3M      0.8 - 2.3 chunk average        16 cells
L5 全球     0.8 - 1.6   1.3M - 5M        <= 0.8    equirectangular      5 cells
```

The split between L3 and L4 is where a cell stops covering a whole pixel, which
is the point at which per-cell rendering stops being meaningful and aggregation
becomes mandatory. L0 through L3 reuse the existing 512/256/128/64 brush texture
LOD cache; L4 and L5 reuse the existing chunk layout and aggregate hierarchy.

`ZoomTierPolicy` selects the tier with six percent symmetric hysteresis so a zoom
sitting on a boundary does not flip the render mode on every wheel notch.

## Brush diameter is clamped per tier

```text
tier      min    suggested    max
L0          1            2     35
L1          1            6    100
L2          1           19    300
L3          1           54    500
L4         32          150    500
L5         64          300    500
```

The minimum matters at L4 and L5: a one-cell brush there edits a real cell but
changes nothing the viewer can see, because one texel of the 1024x512 surface
covers about five cells. The maximum matters at L0 and L1, where an unclamped
500-cell brush would paint nine screens beyond the window in every direction.
Changing tiers snaps the spinbox into the new range.

## Picking

`app/sphere_picker.py` inverts the orthographic projection and then names the
lattice point directly:

1. `screen_to_direction` inverts `software_globe.rotate_point` exactly. The
   sphere radius is passed in rather than recomputed, so the picker always
   matches the radius the caller actually drew with.
2. A direction lies in exactly one base face. Its barycentric weights are a
   3x3 solve against that face's vertices, precomputed as 20 inverse matrices.
3. Rounding those weights to the frequency grid with largest-remainder
   distribution names a CellId through `topology.point_id`.
4. A bounded hill climb over the neighbour graph removes the gnomonic rounding
   error that can appear near a base edge.

The result is the true nearest cell center at every zoom, verified against
brute force over all cells at low frequency and against `cell_center` round trip
at frequency 1004. A pick costs about 26 microseconds, so no seed table or
last-cell cache is needed.

The 2D net editor needed no equivalent work: `FlatNet.nearest_cell` already
resolved a CellId barycentrically at any scale, and only the zoom gate in
`_paint_start` was blocking it.

## Brush cursor

Both editors draw the brush footprint under the pointer. In the sphere viewport
it is the geodesic cap projected as a 48-point polygon, so it foreshortens toward
the limb and shows the true painted extent rather than a nominal circle; samples
behind the limb are dropped instead of folded through the planet. In the net
editor one world unit is one cell, so the footprint is a circle scaled by zoom.
The outline is yellow for paint, red for erase and blue for the undo brush.

## Undo

`app/undo_history.py`. The unit of undo is one stroke.

Before a stroke first writes into a chunk, the chunk's whole cell array is
snapshotted as a packed `uint16` buffer. Packed snapshots are what make the
per-chunk granularity cheaper than per-cell for large strokes:

```text
500-cell-diameter stroke, per cell:   188,000 cells x 6 bytes  = 1.1 MB
500-cell-diameter stroke, per chunk:  729 chunks x 516 bytes   = 376 KB
```

A tuple of boxed Python ints would cost about 9 KB per chunk instead of 516
bytes, which would make the same stroke 6.8 MB. Default depth is 32 strokes with
a 192 MB ceiling.

Undo restores the buffer and marks the chunk dirty, so it reuses the existing
save path and adds no on-disk format. Redo entries are produced when undoing
rather than stored up front, so a stroke that is never undone costs one snapshot.

Two ways to undo:

- `Ctrl+Z` / `Ctrl+Y` reverts or replays a whole stroke.
- The `撤销笔刷` tool reverts each cell under the footprint by exactly one change
  of its own, walking the stack top-down. A cell the newest stroke never touched
  still reverts its own last change rather than jumping to the newest stroke's
  state. Cells with no recorded history are left alone rather than cleared, so
  brushing over never-painted terrain does nothing. The undo brush is itself a
  stroke, so a second pass over the same cells replays what the first pass
  reverted.

Constraints that are enforced rather than documented away:

- Local brush ids are never recycled. Snapshots hold raw `uint16` values that
  reference the map brush table, and other cells may still reference the same
  entry. `set_raw_values` rejects any value naming an id the session does not
  have. An orphaned brush table entry is safe; a freed one would corrupt the map.
- The stack survives a save. Undoing already-saved work legitimately re-dirties
  those chunks and needs another save.
- The stack is cleared when the session changes, because snapshots are only
  meaningful against the brush table they were taken from.
- Undo waits for the stroke queue to drain, the same rule `Ctrl+S` already used.

## Stroke cost at far tiers

Far tiers force a larger brush, and the cost of a stroke is driven by the cell
count, not by the zoom. Two things made that far worse than it needed to be.

**Re-dilation.** `plan_segment` used to re-expand the entire brush disc on every
pointer move and then discard the part already painted. At diameter 500 that was
188,251 cells expanded to yield 1,503 new ones.

**Neighbour cache.** `cell_neighbor_ids_unordered` held 32,768 entries, which is
smaller than any disc past about diameter 210, so the re-expansion could not even
hit cache. That is the cliff between diameter 150 and diameter 300.

Measured cost of one small pointer move during a held stroke, frequency 1004:

```text
diameter    before     incremental    + cache 262144
     150     14 ms          6 ms             7 ms
     300    450 ms         19 ms            19 ms
     500   1520 ms        427 ms            46 ms
```

The incremental form keeps each reached cell's best known graph distance to the
stroke path and only expands a cell when a new segment strictly improves it.
Its relaxation condition is exactly the prune "this cell cannot reach anything
uncovered", so no separate pruning pass is needed. The accumulated covered set is
identical to re-dilation; `tests/test_stroke_incremental.py` asserts that segment
by segment against the old algorithm over random walks.

The first dab of a stroke still expands the whole disc, because it genuinely is
all new: about 1.0 s at diameter 500. That is inherent to a procedural topology
that computes neighbours on demand at roughly 5 microseconds per cell.

## Far view renders the saved surface, not the aggregate mosaic

Until now the native GPU viewport at LOD4 drew one opaque average-color hexagon
per selected aggregate node on top of the textured sphere. That mosaic was
strictly worse than what it covered - one flat color per up-to-16k-cell node
against roughly 5x5 cells per texel of the 1024x512 surface - and a node whose
cache entry did not exist yet rendered as an opaque dark slab. The software
viewport removed exactly this drawing style in v1.0.3; the GPU path had kept it.

The far view now emits an empty instance batch and lets the inset surface sphere
show through, which also means no aggregate node ever needs to be generated for
display. The whole generate-on-view pipeline (budget, progressive fill, pending
counter) is gone from the controller; `ensure_nodes` and its `generate_budget`
remain for the explicit rebuild button, the CLI tool and acceptance.

Painting is live at every LOD in the GPU viewport as well:

- the LOD4 reset patch now reports `editable=True`, and both viewport launch
  paths pass `editable=True`;
- a picked cell that is not in the instance stream is no longer refused - the
  pick resolves an exact CellId from the topology alone and the stroke applies
  to the authoritative Pack session. Only the yellow selection highlight needs
  the instance, and it simply stays off. Far-view strokes report
  "远景笔划已写入 N 格；Ctrl+S 保存后地表纹理更新".
- a standalone GPU window opened after the surface cache was built now receives
  the surface texture at launch instead of showing a blank shaded sphere.

## Replacing a brush and refreshing the library

Brush identity is keyed by path, not by content: rescanning a file that still
sits at the same relative path reuses its uid and only updates the content hash.
The map's `brush_table.json` stores uids, so replacing artwork in place keeps
every painted cell pointing at the same entry and the new image simply takes
over. Deleting a file instead marks the record `missing`, and its cells fall back
to the missing-texture layer without losing their stored state, so restoring the
file brings the artwork back.

Adding a file mints a fresh uid and the image joins its category's random pool
immediately; nothing already painted changes, and the map's brush table only
grows when a brush is actually used. Deleting retires the record. Moving or
renaming is recognised by content, so the uid and every painted cell survive.

Move detection used to require the content hash to be unique on both sides,
which meant relocating several byte-identical brushes in one refresh retired all
of them and minted fresh uids - every cell painted with any of them turned into
missing texture. Unmatched files are now paired one for one against retired
records of the same hash. Because the files are identical, which record claims
which path cannot change what is drawn. This reverses a deliberate "do not guess
an ambiguous move" rule: the trade is that a genuinely new file that happens to
be byte-identical to a deleted one now inherits its uid instead of both cells and
file going their separate ways.

Two things did not follow that change through:

- A GPU texture layer is keyed by uid *and* content hash, while instances hold a
  resolved layer index. `update_records` only swapped the dictionary, so cells
  already in the stream kept rendering the previous artwork until their chunk
  happened to be evicted and re-added, and the superseded layer was never
  released. It now detects a content or availability change, resets the stream so
  every visible cell re-resolves, and reports that to the caller, which forces a
  surface rebuild and a redraw in both viewports.
- The LOD cache path embeds the content hash, so the previous four PNGs per brush
  were stranded forever. `BrushLodCache.prune_superseded` removes the other hash
  directories for that uid after a regeneration. Nothing refers to them: layer
  keys and manifests always derive from the record's current hash. The trade is
  that reverting a brush to earlier artwork regenerates its cache instead of
  finding it still on disk.

## One ladder drives both the label and the render

The tier table used to be advisory. `production_streaming` selected the GPU level
through `BrushLodPolicy` thresholds fed by its own private cell estimate:

```python
estimate = hexagon_count * aspect_factor / (2.0 * zoom * zoom)
```

That runs about 1.74x higher than the spherical-cap figure `zoom_tiers` computes
and the interface displays, so the two drifted apart by one to two levels. On a
1180x720 viewport:

```text
zoom     tier says      GPU actually rendered
 5.2     L3 64x64       LOD4, far-view surface only
13.4     L2 128x128     LOD4, far-view surface only
21.5     L2 128x128     LOD3
55.0     L1 256x256     LOD2
```

Per-cell brush artwork therefore did not appear until roughly zoom 15-21 even
though the status line had been claiming per-cell texturing since 4.5.

`ProductionLodController` now selects straight from `ZOOM_TIERS` through a
`ZoomTierPolicy`, and `_estimated_visible_cells` delegates to
`zoom_tiers.visible_cells_estimate`, so there is a single definition of what each
zoom range means. `BrushLodPolicy` still owns the texture sizes.

Reaching down to the L3 floor needed more streaming headroom: measured on the
real f1004 planet, the L3 hysteresis edge wants 1,727 chunks, about 34 MB of
instance data, against the old `maximum_visible_chunks=1200`. The budget is now
2048. Measured on the real map, per-cell textures now start at zoom 5.5 instead
of 15.6, and a warm wheel notch costs 17 ms instead of 107 ms.

A per-cell tier that still exceeds the budget - a much larger window sees
proportionally more chunks - now falls back to the far-view surface for that
frame instead of raising `ProductionStreamingError`. The exception only ever
reached a status line, so the viewport would freeze on stale content.

## The frozen-LOD regression

At one point `production_editor.py` gained `logging.info` diagnostics in the
view worker without gaining `import logging`. Every GPU view request then died
on a NameError before reaching `update_view`, the error surfaced only in the
WGL window title, and the viewport stayed at LOD4 forever: zooming in never
streamed a single textured cell. The same drift also referenced
`GpuBatchResetPatch` without importing it, which - once the first bug was fixed -
would have killed the Tk poll chain on the first LOD switch instead.

Both imports are fixed, and `tests/test_module_names.py` now AST-scans every
module under `app/` for loads of names that are neither imported nor defined,
because `compileall` cannot catch this class of error.

## Far-view zoom cost

At LOD4 the far view is drawn from the aggregate hierarchy, and `update_view`
called `ensure_nodes` for the whole selection inline. A whole-planet view selects
several hundred nodes, and summarising one node reads every Pack chunk beneath it
(about 61 chunks for a 16,000-cell node), so a single wheel notch on a cold cache
blocked the view thread for tens of seconds. Measured per notch, frequency 1004,
900x900 viewport:

```text
zoom     before      budgeted    + manifest memo
0.93     15.3 s        336 ms          330 ms
1.08     43.3 s        455 ms          440 ms
1.45     61.1 s        625 ms          610 ms
1.95     51.7 s        514 ms          500 ms
3.04      4.1 s        411 ms          400 ms
warm     174 ms        174 ms          107 ms
```

Two changes:

- `ensure_nodes` takes a `generate_budget` (8 per view update). Nodes left
  unbuilt come back as `pending_count`, `ProductionStreamFrame.has_more` goes
  true, and the editor re-issues the same view request so the remainder fills in
  over later frames. This is safe because `build_aggregate_proxy_batch` already
  renders a node with no summary as the empty layer, so the far view appears
  immediately and sharpens instead of stalling. The explicit cache rebuild and
  `build_aggregate_lod.ps1` pass no budget and still build everything.
- `AggregateLodCache.existing` memoises the parsed manifest on the file's mtime
  and size. A far view resolved several hundred manifests twice per frame, once
  to decide what to build and again to build the proxy batch. Invalidation
  deletes the manifest, so a deleted or rewritten node misses the memo.

### Why the far view can sit on blank blocks

A node with no summary renders as the empty layer, so a budget that is small
relative to the number of missing nodes shows blank blocks until the fill
catches up. On a real f1004 planet one whole-planet view selects 971 nodes; with
255 of them missing the fill originally took 41 s, and any camera movement in
that window restarts it against a different selection.

Building a single node was 344 ms, and almost none of that was the work itself:

```text
per node        before    after
fsync x2         86 ms       0 ms   derived cache, no durability requirement
pack opens       55 ms       2 ms   632 opens -> one per pack file
python loop      90 ms      45 ms   values hashed in runs, not two bytes at a time
reads/hash       113 ms     21 ms
total           344 ms      68 ms
```

- `atomic_write_png` grew a `durable` flag and the aggregate manifest writer
  dropped its fsync. The cache is regenerated from the Pack map, so losing it to
  a crash costs a rebuild, not data.
- `SphereMapStore.read_chunk_values_many` groups chunks by pack file and reads
  each pack once. Summarising one node was opening and re-validating a pack file
  632 times.
- The digest accumulates cell values into one buffer and flushes it before any
  brush identity is mixed in, so the hashed byte stream is unchanged and every
  node already on disk stays valid. `tests/test_aggregate_digest.py` asserts the
  signatures against the original per-cell implementation.
- `update_view` memoises the visibility query and the aggregate selection on the
  camera, since a progressive fill repeats the identical view many times.

The GPU status line now reports how many nodes are still pending, so a partially
built far view reads as progress rather than as a broken render.

Prebuilding still gives the best far view, because a budgeted fill only sharpens
as fast as frames arrive. For one viewpoint:

```powershell
pwsh.exe -File apps/tools/build_aggregate_lod.ps1 -MapName planet -Zoom 1.0
```

For the whole hierarchy, roughly nine minutes for 8,191 nodes:

```powershell
pwsh.exe -File apps/tools/build_aggregate_lod.ps1 -MapName planet -All
```

## What is still deferred

L4 and L5 paint correctly but their feedback still waits for the next save,
because the far view is the saved 1024x512 surface. Making those tiers update
live needs the chunk-average and per-cell bitmap rasterisers, which must be
rendered in a worker into one `PhotoImage` rather than as canvas items: 7,000
`create_polygon` calls is already near the Tk limit, and L3 would need 170,000.
`detail_cell_limit` is the guard that keeps the per-cell path from being
attempted beyond its budget.
