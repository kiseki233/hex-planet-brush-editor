# Hex Planet Brush Editor v1.3.1

## Final implemented scope

The project root contains only `apps/` and `art/`.

The editor now implements the complete software-side path defined by the detailed design:

- Unicode brush categories, stable 128-bit brush UIDs, missing-resource recovery, and the 4095-brush limit.
- Strict 512×512 RGB/RGBA PNG validation.
- Built-in manual-scale image cropper for exporting one or many 512×512 PNGs to `art/data/`.
- Four-pixel copied edge padding, nearest-neighbor sampling, and six shader UV rotations.
- Stable uint16 cell states and per-map brush tables.
- Procedural frequency-1004 topology with 10,080,162 stable CellIds and 12 hidden pentagons.
- 39,060 connected logical chunks and compact hierarchical visibility indexes.
- Indexed Pack storage, independent compression, CRC32 isolation, atomic dirty-chunk save, compaction, and interrupted-transaction recovery.
- Background Pack reads and visible-only dense GPU instance streaming.
- Automatic LOD0–LOD4 selection with hysteresis.
- Multi-level aggregate far-view hierarchy and regenerable aggregate node caches.
- Ref-counted GPU texture-array residency, grace-period release, and deterministic slot reuse.
- Frame-budgeted visible chunk additions/removals with center-first priorities.
- Automated seam/UV validation and a 256-distinct-LOD0 texture load test.
- Windows/Tk/PowerShell/WGL diagnostics and a one-click final acceptance report.

## Directory contract

```text
HexPlanetBrushEditor_v1.3.1/
├─ apps/
└─ art/
   ├─ brushes/
   ├─ data/
   └─ maps/
```

No launcher, cache, log, configuration file, or document is placed beside `apps/` and `art/`.

## Start on Windows

Double-click:

```text
apps/start.bat
```

The launcher prefers PowerShell 7 (`pwsh.exe`) and falls back to Windows PowerShell. The source build requires Python 3.10 or newer with tkinter. It does not install or modify the environment automatically.

## Import brushes

Place 512×512 PNG files under any category hierarchy:

```text
art/brushes/
├─ 海洋/
├─ 城市/
└─ 地形/森林/
```

Then select `刷新笔刷库`. Invalid dimensions or damaged PNG files are reported rather than silently resized.


## Built-in brush image cropper

Select `图片裁剪器` above the brush tree. Opening a source PNG only loads a preview; it never exports automatically. The cropper keeps a fixed 512×512 output grid while the source image can be scaled and moved underneath it.

- Mouse wheel: change the actual source-image scale around the cursor.
- Right drag: move the source image.
- Middle drag: navigate the workspace without changing the export.
- Left drag: select one or many output cells.
- Yellow ruler: exactly one output cell, labelled as 3 km for manual scale-bar alignment.
- Dashed hexagon: preview of the area that the spherical cell geometry will display.

Every selected cell is written as one 512×512 RGBA PNG under the chosen `art/data/` folder. The cropper does not register or copy those files into the brush library. Existing files are not overwritten; a new batch suffix is selected automatically. A source region outside the image becomes transparent, and completely outside cells are rejected. See `IMAGE_CROPPER.md`.


## 2D unfolded map editor

Select `打开 2D 展开编辑器` from the main editor. The twenty icosahedron base faces are unfolded into a flat triangular net while keeping the same CellIds and the same `art/maps/planet/` Pack data. The sphere and flat views are two views of one map. Seam cells can appear in more than one cut face, but every copy edits the same CellId.

Use the mouse wheel to zoom, right-drag to pan, and hold the left mouse button to paint continuously with the brush group and 1–500-cell diameter selected in the main window. At overview scale the saved surface is projected over the net; at detailed scale the real editable cell polygons are shown. See `FLAT_MAP_EDITOR.md`.

## Random brush groups and continuous strokes

The selectable brush is a category folder rather than one fixed image. For example,
selecting a `城市` folder containing six PNG files makes every newly covered cell choose
one of those six images independently and choose one of the six 60-degree rotations. The
chosen UID and rotation are saved normally and remain stable after reopening the map.

The brush diameter is adjustable from 1 through 500 cells. Hold the left mouse button and
drag to paint continuously. The editor connects successive sampled CellIds through the
sphere topology before applying the circular footprint, so fast movement does not leave
gaps. A cell is randomized only once during one held stroke; a later stroke can completely
overwrite and rerandomize it. Existing content is never blended.

Large strokes run on a dedicated worker, coalesce stale mouse-move commands, batch writes
by Pack chunk, and delay save until the active stroke is complete. See
`apps/BRUSH_GROUP_STROKES.md`.

## Main planet editor

The application now opens directly into the one complete planet map. The legacy 16×16 local prototype is no longer the startup screen and there are no new/open map-name controls in the normal interface.

The authoritative map is fixed at:

```text
art/maps/planet/
```

If it does not exist, the program creates it automatically. If it already exists, the same planet is opened automatically. Legacy local prototype folders are ignored rather than deleted.

Main-view controls:

```text
Left mouse drag   continuously paint/erase with the selected 1–500-cell brush
Right mouse drag  rotate the planet
Mouse wheel       zoom the planet
Ctrl+S            save the planet after queued stroke work completes
Ctrl+Z / Ctrl+Y   undo or redo one complete stroke
```

Painting is available at every zoom level. The pointer position is resolved to a
CellId analytically rather than by hit-testing rendered polygons, so far tiers
lose feedback resolution but not edit precision. See `ZOOM_TIERS_AND_UNDO.md`.

The central canvas always represents the complete sphere. On Windows, the central area automatically starts an embedded WGL/OpenGL viewport. The separate GPU button can restart that viewport. If WGL startup fails, the editor keeps the optimized software globe as a fallback.

The editor prepares or reads:

```text
art/maps/.topology/ico_dual_f1004_v2/
├─ production.json
├─ production_chunks.idx
├─ production_visibility.json
└─ production_visibility.idx
```

Production constants:

```text
frequency            = 1004
cells                = 10,080,162
pentagons            = 12
editable hexagons    = 10,080,150
logical chunks       = 39,060
visibility nodes     = 8,191
materialized cells   = 0
materialized tris    = 0
```

The procedural topology calculates centers, ordered neighbors, and dual corners only for requested CellIds.

## Six-tier zoom ladder

```text
L0 原图   zoom >= 110    <= 290 cells       brush 1–35
L1 精细   38 – 110       290 – 2.4k         brush 1–100
L2 常用   13 – 38        2.4k – 20k         brush 1–300
L3 区域   4.5 – 13       20k – 170k         brush 1–500
L4 地块   1.6 – 4.5      170k – 1.3M        brush 32–500
L5 全球   0.8 – 1.6      1.3M – 5M          brush 64–500
```

Every tier is editable. The brush diameter is clamped into the active tier's
range on every tier change: the minimum keeps a far-view stroke larger than the
feedback granularity, and the maximum keeps a near-view stroke from covering
several screens beyond the window. See `ZOOM_TIERS_AND_UNDO.md`.

## Undo

One stroke is one undo step. Chunk cell arrays are snapshotted as packed uint16
buffers before their first modification within a stroke, so a 500-cell-diameter
stroke costs 376 KB rather than 1.1 MB. The default depth is 32 strokes.

`Ctrl+Z` and `Ctrl+Y` revert or replay a whole stroke. The `撤销笔刷` tool reverts
each cell under the footprint by one change of its own. Undo never recycles local
brush ids, survives a save, and is cleared when the session changes.

## Automatic production LOD

The GPU production path selects the display level from the estimated visible cell count and uses hysteresis:

```text
LOD0  512×512 effective brush content   zoom >= 110
LOD1  256×256                           38 – 110
LOD2  128×128                           13 – 38
LOD3   64×64                            4.5 – 13
LOD4  saved 1024×512 surface sphere     0.8 – 4.5
```

The level is selected from the same six-tier table the interface displays, so the
reported tier and the rendered detail always agree. Editing is live at every
level; at LOD4 the result appears after the next save, when the surface texture
is rebuilt. A view whose chunk count exceeds the streaming budget degrades to the
surface for that frame rather than failing.

At LOD0–LOD3, only candidate Pack chunks enter the dense instance buffer. At LOD4, individual cells leave the active buffer and selected aggregate hierarchy nodes represent the visible planet.

## GPU residency and streaming

- One ordinary hexagon is one 80-byte exact-corner instance record.
- Entering chunks append instances.
- Leaving chunks use deterministic swap-remove.
- View changes are applied in bounded batches, with screen-center chunks loaded first.
- One brush UID/content hash/LOD combination occupies one texture-array layer regardless of repeated cell use.
- Unreferenced brush layers are released after a short grace interval.
- Released texture slots are reused instead of growing the array indefinitely.
- New or recycled layers are uploaded with `glTexSubImage3D`.
- A visible cell edit updates one 80-byte exact-corner record with `glBufferSubData`.
- Pack reads used for rendering do not remain in the authoritative map session cache.

## Multi-level aggregate cache

Derived caches are stored under:

```text
art/maps/<map-name>/lod/aggregate_v1/
```

Only selected far-view nodes need to be generated. A full-tree prebuild is optional. A saved chunk edit invalidates the affected leaf and every aggregate ancestor, while unrelated nodes remain reusable.

Command-line selected-view build:

```powershell
pwsh.exe -File apps/tools/build_aggregate_lod.ps1 `
  -MapName planet_production_f1004 `
  -Zoom 1.0
```

## Production map build and verification

```powershell
pwsh.exe -File apps/tools/build_production_plan.ps1 `
  -Frequency 1004 `
  -TileSide 16 `
  -MapName planet_production_f1004 `
  -CreateMap `
  -VerifyMap
```

A complete blank production map contains 39 Pack files and 39,060 independently readable chunks.

## Final acceptance

GUI:

```text
Main planet editor → 运行最终验收与 Windows 诊断
```

Command line:

```powershell
pwsh.exe -File apps/tools/run_acceptance.ps1
```

It checks the root contract, brush catalog, frequency-1004 indexes, complete frequency-1004 Pack map, LOD0–LOD4 streaming, Pack save/reopen/CRC/compaction, aggregate caches, shared-edge/UV seams, 256 different LOD0 texture layers, and Windows/WGL capabilities.

Reports are written to:

```text
apps/runtime/logs/acceptance_report.json
apps/runtime/logs/acceptance_report.txt
```

## Windows diagnostic

```powershell
pwsh.exe -File apps/tools/run_windows_diagnostic.ps1
```

The diagnostic records Python, tkinter, PowerShell, launcher files, Windows GPU inventory, WGL/OpenGL version, renderer/vendor, maximum texture size, and maximum texture-array layers.

## Platform boundary

All topology, Pack, LOD, aggregate, texture residency, streaming, edit/save, seam, stress, diagnostics, and Tk integration code is implemented and tested in the available Linux environment.

The Win32/WGL window cannot be truthfully marked as executed on a real Windows GPU from this environment. Run the included Windows diagnostic and final acceptance on the target PC. That is the only remaining external validation boundary; it is not an unimplemented software subsystem.

## v1.0.3 sphere-geometry correction

The startup canvas no longer draws aggregate hierarchy nodes as detached regular hexagons.
At whole-planet distance it renders one continuous shaded sphere. After sufficient zoom it
projects the six actual procedural dual corners of each visible cell. Adjacent cells therefore
reuse the same two projected edge endpoints and can be edited directly in the main canvas.

The native OpenGL path also changed from a center/tangent/fixed-radius approximation to an
80-byte record containing six exact spherical corner vectors plus texture layer and rotation.
A slightly inset continuous sphere mesh is rendered below the cell layer, so the limb and far
LOD never expose the window background through approximation gaps.

## v1.0.3 software far-surface and drag update

The main Tk globe now builds a regenerable `1024×512` equirectangular surface from the
saved Pack map. Each production chunk contributes its saved cell-state average and brush
representative color. The texture is projected back onto the rotating orthographic sphere,
so painted ocean, land, city, and other regions remain visible after zooming out.

The cache is stored at:

```text
art/maps/planet/lod/software_surface_v1/
├─ planet_1024x512.png
└─ surface.json
```

It is rebuilt automatically after a successful map save and after brush content changes.
Deleting it does not remove map data.

Right-drag rendering now uses a latest-request-only persistent process. During the drag it renders a
one-quarter-resolution temporary surface and scales it to the canvas; after release it replaces
that preview with the normal-quality image and restores exact cell polygons. Heavy visibility
queries, Pack reads, and shared-corner polygon generation are skipped while the button is held.
The detailed view also retains the current visible Pack chunks instead of rereading them for
every motion event.


## v1.0.4 main-viewport latency correction

The Windows main window now embeds the native WGL renderer inside the central Tk viewport.
The toolbar, brush catalog, Pack session, and save logic remain owned by Tk/Python, while
rotation and zoom are handled by GPU uniforms. The 1024×512 saved-map surface is uploaded
as a normal 2D texture and sampled by the sphere fragment shader. LOD0–LOD3 continue to use
exact-corner instanced hexagons.

The software fallback was also corrected and isolated from Tk by a persistent multiprocessing renderer:

- drag scheduling changed from 33 ms to 16 ms;
- drag preview changed to four-times downsampling;
- full-quality rendering is delayed for 100 ms after input stops;
- a newer earlier deadline can cancel a pending Tk draw callback;
- monotonically increasing render generations discard stale results before `PhotoImage` creation;
- the Canvas surface image is persistent and unrelated items are no longer rebuilt with `delete("all")`.

Native streaming updates no longer call `glBufferData` for every patch. The VBO keeps a
power-of-two capacity, swap-remove/added/changed slots use `glBufferSubData`, and a full VBO
reallocation occurs only when capacity grows or the LOD batch is reset. Texture-array uploads
are limited per rendered frame, and tiny camera changes no longer trigger a CPU visibility
request.

## v1.1.0 random brush groups and continuous painting

A brush category directory is now the selectable brush unit. Every active PNG in that
category and its descendants participates in an independent random choice for each newly
covered cell. The cell also receives an independent random rotation from the six supported
60-degree directions. The selected UID and rotation are persisted normally and do not change
when the map is reopened.

Painting fully replaces the old cell state. Brush diameter is adjustable from 1 to 500 cells.
Holding the left button starts one continuous topology-aware stroke: successive sampled cells
are connected through neighbors and the selected footprint is applied along the whole path,
so fast cursor motion does not leave gaps. A cell is randomized only once in one held stroke;
a later stroke may overwrite and randomize it again.

Large strokes run on a dedicated worker, stale move positions are coalesced, affected cells are
written in logical-chunk batches, and save waits for queued stroke work. See
`BRUSH_GROUP_STROKES.md` for the exact behavior.


## v1.3.1 flat numbered cropper output

The cropper writes every selected 512×512 tile directly into the single `art/data/` directory using a global numeric sequence: `001.png`, `002.png`, `003.png`, and so on. It creates no category subfolders and does not register cropped files as brushes. After manual review, move or copy accepted images into `art/brushes/<category>/` and refresh the brush library; the brush catalog then assigns persistent UIDs independent of the numeric filenames.
