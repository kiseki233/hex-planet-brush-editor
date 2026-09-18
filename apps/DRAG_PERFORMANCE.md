# Main Viewport Drag Performance v1.3.1

## Windows primary path

On Windows, the central Tk canvas hosts a native `WS_CHILD` WGL/OpenGL 3.3 viewport. Tk still owns the brush catalog, tool state, authoritative map session, dirty chunks, and Pack saves. The child viewport owns real-time sphere rotation, zoom, distant-surface sampling, and exact-corner instanced hexagon drawing.

Mouse rotation changes only `yaw`, `pitch`, and shader uniforms every rendered frame. CPU visibility requests are submitted only when the accumulated view change reaches one of these thresholds:

```text
yaw delta      >= 0.01 rad
pitch delta    >= 0.01 rad
zoom ratio     >= 1.02
size changed   always submit
```

## Software fallback path

The fallback remains available when WGL cannot start or on non-Windows systems.

- Tk redraw scheduling is 16 ms instead of 33 ms.
- Drag and wheel interaction use one-quarter-resolution preview rendering.
- Full-quality rendering waits 100 ms after input stops.
- A newer earlier Tk deadline cancels a later pending callback.
- Render generations reject stale completed frames before `PhotoImage` creation.
- The software renderer runs in one persistent spawned process, so pure Python pixel loops no longer hold the Tk process GIL.
- Request and result queues are bounded and keep only the newest useful frame.
- The Canvas image, outline, and overlay items are persistent; drag updates do not call `canvas.delete("all")`.

## Native streaming upload path

- The instance VBO keeps power-of-two capacity.
- Added, changed, and swap-moved slots use `glBufferSubData`.
- Full VBO allocation happens only when capacity grows or a full LOD batch reset is received.
- Texture-array uploads are queued and limited to 2 layers per frame while dragging and 8 layers per frame otherwise.
- Updated distant-surface textures are pushed into the existing GPU viewport without restarting it.

## Validation boundary

The process renderer, Tk fallback interaction, stream-delta logic, Pack behavior, and all cross-platform code paths are executed by automated tests. The embedded `WS_CHILD` WGL path must still be run on the target Windows GPU because the build environment is Linux.
