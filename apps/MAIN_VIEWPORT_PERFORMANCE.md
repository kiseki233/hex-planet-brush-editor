# Main viewport performance implementation

## Windows primary path

The main Tk window embeds a native `WS_CHILD` WGL/OpenGL viewport in the central canvas area.
Tk owns the brush catalog, tool state, authoritative map session, and Pack writes. The native
child owns interactive rendering and input.

Interactive rotation and zoom update only `uYaw`, `uPitch`, and `uScale`. The saved-map
`1024×512` equirectangular surface is uploaded as `GL_TEXTURE_2D` and sampled by the sphere
fragment shader. Detailed cells remain `glDrawArraysInstanced` records containing the six exact
spherical dual corners.

If WGL startup fails, the child closes and the existing software globe becomes visible again.

## Software fallback

The fallback keeps the same map behavior but uses a bounded low-resolution preview while input
is active:

- 16 ms Tk draw scheduling;
- 4× reduced render dimensions with 4×4 blocks during interaction;
- latest generation only;
- 100 ms full-quality debounce;
- persistent Canvas surface image and tagged overlay/detail objects.

The Python worker cannot cancel a frame after it has entered the pure-Python loop, but stale
frames are never converted into Tk images and only the newest pending request starts next.

## Native stream updates

The instance VBO is allocated with power-of-two capacity. Normal stream patches apply
swap-remove/add/change deltas and upload only affected 80-byte slots with `glBufferSubData`.
`glBufferData` is reserved for capacity growth and complete LOD batch resets.

Texture uploads are queued by layer and drained with a per-frame budget: two layers while the
right button is held, otherwise eight. Recycled layers keep placeholder storage until their
new pixels are uploaded.

CPU visibility requests retain the 80 ms rate limit and additionally require at least one of:

- yaw delta >= 0.01 radians;
- pitch delta >= 0.01 radians;
- zoom ratio delta >= 2%;
- viewport size change;
- an explicit follow-up request because the prior patch has more work.

## Verification boundary

All Python logic, shader source, patch contracts, software fallback behavior, and integration
startup were tested in the available Linux/Xvfb environment. A real embedded WGL child window
still requires execution on the target Windows GPU and is reported as an external validation
boundary rather than silently marked as passed.
