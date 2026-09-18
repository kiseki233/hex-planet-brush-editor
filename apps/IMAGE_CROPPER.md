# Brush Image Cropper v1.3.1

The built-in cropper creates fixed 512×512 PNG source tiles for later manual sorting.

## Output boundary

All cropped files are written directly into one flat directory:

```text
art/data/
├─ 001.png
├─ 002.png
├─ 003.png
└─ ...
```

The cropper never creates category subfolders and never registers these files as brushes. It scans existing numeric PNG names and continues from the highest number. Names use at least three digits, so numbering continues as `999.png`, `1000.png`, `1001.png`, and so on. Existing files are never overwritten.

## Workflow

1. Open one 8-bit RGB or RGBA PNG source image.
2. The source is loaded only for preview; no file is generated automatically.
3. Use the mouse wheel to change the actual source-image scale.
4. Right-drag to move the source image beneath the fixed crop grid.
5. Use the yellow one-cell guide to align a 3 km scale bar in the source image.
6. Left-drag across one or more grid cells.
7. Export. Every selected cell becomes the next numbered 512×512 RGBA PNG in `art/data/`.
8. Review and classify the results yourself.
9. Move or copy accepted files into `art/brushes/<category>/`.
10. Refresh the brush library. Only then does the catalog assign each file a persistent 128-bit brush UID.

The numeric filename is not the brush UID. Map references use the persistent UID stored in `art/brushes/.catalog/brushes.json`. A uniquely identifiable file moved between brush categories preserves its UID; two simultaneous copies are separate brush entries and receive separate UIDs.

The dashed hexagon is a preview of the part visible after the square PNG is mapped to a map cell. The exported file remains square because the renderer clips it with the real spherical hexagon geometry.

## Controls

- Mouse wheel: scale the source image around the cursor.
- Right drag: move the source image.
- Middle drag: move the workspace view.
- Left drag: select a rectangular group of output tiles.
- Workspace percentage: change only the editor view scale; it does not change exported pixels.

The cropper rejects selections containing cells completely outside the source. Partially covered cells can be exported with transparent pixels after confirmation.
