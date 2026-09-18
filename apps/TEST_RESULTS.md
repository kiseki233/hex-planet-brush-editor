# Hex Planet Brush Editor v1.3.1 Test Results

## Scope

v1.3.1 changes the cropper staging workflow without changing the shared spherical/2D map data model.

Implemented changes:

- Every crop is written directly into the single flat `art/data/` directory.
- No crop-category subfolders and no source-name prefixes are created.
- Files use one global numeric sequence: `001.png`, `002.png`, `003.png`, ... `999.png`, `1000.png`, and so on.
- Existing numeric PNG files are detected and never overwritten.
- Non-numeric files in `art/data/` do not disturb the numeric sequence.
- Cropped files remain ordinary staging data and receive no brush UID.
- After a file is moved or copied into `art/brushes/<category>/`, refreshing the brush library assigns a persistent 128-bit UID.
- A uniquely identifiable brush moved between category folders keeps the same UID.
- Two simultaneous copies remain separate brush entries and receive separate UIDs.

## Automated tests

All 35 test modules were run in isolated Python processes so the large production-topology caches do not accumulate between stress modules.

```text
Automated tests: 153 passed
Failures:       0
```

New and updated tests cover:

- flat `art/data/001.png`, `002.png`, ... export;
- continuation from the highest existing numeric filename;
- ignoring non-numeric PNG names while choosing the next number;
- numbering beyond `999.png`;
- no overwrite across multiple export runs;
- 512×512 RGBA output and spatial source preservation;
- no UID while the image remains in `art/data/`;
- UID assignment after moving the image into `art/brushes/城市/`;
- UID preservation after moving the same brush into another category;
- all previous brush catalog, cropper, 2D map, spherical map, Pack, LOD, GPU streaming, seam, topology, and diagnostic tests.

## GUI check

Tk/Xvfb cropper startup was executed against an empty project layout.

```text
cropper_gui_smoke=passed
next_output=001.png
category_folder_control=absent
```

## Final acceptance

The full texture-stress acceptance run completed:

```text
Passed:        12
Failed:        0
Needs Windows: 1
```

The cropper acceptance exported `001.png` and `002.png`, moved `001.png` into a temporary brush category, refreshed the brush catalog, and verified that a persistent UID was assigned. The texture-array stress used 256 distinct LOD0 brushes. The remaining target-only item is the real Windows embedded WGL/OpenGL viewport.
