# Random Brush Groups and Continuous Strokes v1.3.1

## Brush group selection

A category directory is one brush group. Selecting `城市` uses every active PNG in
`城市/` and its descendant directories. Selecting a displayed image selects the same
parent group; individual files are not treated as a separate manual-rotation tool.

For each newly covered cell, the editor independently chooses:

```text
one active image from the selected group
one rotation from 0°, 60°, 120°, 180°, 240°, 300°
```

The chosen brush UID and rotation are written into the normal map cell state. Reloading
the map does not randomize existing cells again.

## Complete replacement

Painting does not blend with the old state. Every affected cell receives a complete new
16-bit state. Painting over an occupied city, ocean, or terrain cell fully replaces the
previous local brush ID and rotation. Erase writes zero.

## Diameter

The UI accepts a diameter from 1 through 500 cells. The brush footprint is a connected
sphere-topology disk around the stroke centerline. The 12 reserved pentagons participate
in topology traversal but are never written.

Large footprints are calculated on the dedicated brush-stroke worker. Mouse moves for
the same active stroke are coalesced to the latest position, and save requests wait until
the queued stroke has completed.

## Continuous painting

```text
left button down  start one stroke
left button move  connect the previous and current CellId through topology neighbors
                  and dilate that centerline by the selected diameter
left button up    finish the stroke
```

Fast mouse motion therefore cannot leave unpainted gaps between sampled positions. A
cell is randomized at most once inside one stroke. Releasing the button and starting a
new stroke allows the same cell to be randomized and overwritten again.

## Pack and GPU updates

Assignments are grouped by logical chunk. Each affected chunk is loaded once and marked
dirty once. Visible GPU instances are updated as one stream patch, with new texture-array
layers uploaded only for group images that become visible.
