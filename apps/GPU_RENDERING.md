# GPU Batch Rendering and Editing v1.3.1

## Authority boundary

The Tk application owns `SphereMapSession`, brush tables, dirty chunks, and Pack writes. The native Windows OpenGL 3.3 viewport is a rendering and input client. It sends immutable view/edit/save requests and receives render patches.

## Main-window embedding

The Windows main window creates the renderer as a native `WS_CHILD` inside the central Tk canvas. Resizing the canvas resizes the child viewport. If WGL creation fails or the child closes, the application falls back to the software globe renderer without changing map data.

## Exact instance structure

Each ordinary hexagon uses one 80-byte record:

```text
corner 0..5     18 × float32
texture layer    1 × float32
rotation         1 × float32
```

The vertex shader selects the six real shared dual corners directly. Adjacent cells therefore receive identical edge endpoints instead of independently estimated regular hexagons.

## Rendering

- One shared 18-vertex triangle-fan selector mesh.
- `glDrawArraysInstanced` for the active dense stream.
- `GL_TEXTURE_2D_ARRAY` for brush LOD textures.
- A normal `GL_TEXTURE_2D` for the saved-map distant surface.
- Nearest filtering and clamp-to-edge for brush textures.
- Four-pixel padding excluded by shader UV scaling.
- UV rotation by `rotation × 60°` in the shader.
- `gl_InstanceID` selection highlighting.
- A continuous sphere mesh under the detail cells.

## Incremental VBO updates

The instance VBO uses power-of-two capacity. A stream patch applies exact swap-remove semantics in CPU memory and uploads only added, changed, or moved slots through `glBufferSubData`. `glBufferData` is used only when capacity grows or a complete LOD reset is installed.

## Texture upload budget

Released texture layers lose their old key mappings and may be reused. New stream texture payloads are queued and uploaded over multiple frames:

```text
right-drag active    2 layers/frame
not dragging         8 layers/frame
```

A capacity increase reallocates the array once and restores all resident layers. Updated 1024×512 distant-surface data is uploaded to the existing sphere texture through `GpuSurfaceTexturePatch`.

## Controls

```text
Left click       edit selected cell
Right drag       rotate sphere
Mouse wheel      zoom
P / E            paint / erase
R                next rotation
Ctrl+S           save
Esc              close child viewport and use fallback
```

## Diagnostic

The WGL diagnostic creates a native context, loads the real renderer functions, compiles both shader programs, and reports OpenGL version/vendor/renderer and texture limits. It is available from the final acceptance window and `run_windows_diagnostic.ps1`.
