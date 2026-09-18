# 2D Icosahedron-Net Map Editor v1.3.1

The 2D editor displays the same authoritative `art/maps/planet/` map as the spherical editor. It does not create a second map or export/import a flattened bitmap.

## Layout

The sphere is unfolded into twenty equilateral base-face triangles. The top and bottom poles and one longitudinal cut are duplicated where the net is cut. Duplicated seam cells keep the same stable CellId, so painting either copy changes the same spherical cell.

This projection follows the frequency-1004 icosahedral topology directly. It avoids the severe polar stretching of a longitude/latitude rectangle and preserves the triangular lattice inside every base face.

## Controls

- Mouse wheel: zoom around the cursor.
- Right drag: pan the flat workspace.
- Left drag: continuously paint or erase with the brush group and 1–500-cell diameter selected in the main window.
- Ctrl+S: save the same planet Pack map.

At overview scale the editor renders the saved planet surface over the twenty triangles. After zooming to about nine pixels per cell, real editable cells are displayed. Seam copies share one CellId and one saved state.
