# Native GPU Editing Contract

## Ownership rule

Only the Tk main thread may mutate map state or write Pack files.

The native OpenGL thread may:

- read its immutable GPU batch snapshot;
- perform screen-to-sphere picking;
- submit edit/save requests;
- receive cell and status patches;
- update OpenGL instance and texture resources.

It may not directly call `SphereMapSession.set_cell()` or `SphereMapStore.save()`.

## Bridge messages

Native-to-Tk:

- `GpuEditRequest`: CellId plus a snapshot of tool, brush UID, path, and rotation.
- `GpuSaveRequest`: request to save the current dirty map session.

Tk-to-native:

- `GpuCellPatch`: confirmed CellId, texture key, rotation, optional new RGBA texture payload, and status text.
- `GpuStatusPatch`: save or error status without a cell-resource change.

## Consistency behavior

- A failed authoritative edit does not update the GPU instance.
- A successful edit marks the owning chunk dirty before the GPU patch is returned.
- New map creation, opening another map, topology regeneration, editor shutdown, or brush-catalog refresh closes the existing GPU bridge.
- Tk painting pushes the same cell patch to the native window, keeping both views synchronized.
- Native painting updates the Tk view on the next main-thread poll.
- Pack data remains unsaved until either save surface performs `Ctrl+S` or the save button is used.
