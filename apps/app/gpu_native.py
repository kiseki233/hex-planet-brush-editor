from __future__ import annotations

import ctypes
import logging
import math
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .gpu_batch import (
    EMPTY_LAYER_KEY,
    MISSING_LAYER_KEY,
    GpuRenderBatch,
    GpuTextureLayer,
    INSTANCE_STRUCT,
    build_texture_layer_payload,
    unit_hex_mesh_bytes,
)
from .gpu_memory import (
    DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
    budgeted_texture_layer_limit,
    next_texture_capacity,
    texture_layer_bytes,
)
from .gpu_edit import (
    GpuBatchResetPatch,
    GpuCellPatch,
    GpuEditBridge,
    GpuEditError,
    GpuStatusPatch,
    GpuSurfaceRegionPatch,
    GpuSurfaceTexturePatch,
    GpuStreamPatch,
    GpuTextureUpload,
    SphericalCellPicker,
)
from .topology import DualTopology
from .png_pixels import PixelImage, read_png_pixels
from .zoom_tiers import drag_radians_per_pixel
from .i18n import t

VERTEX_SHADER_SOURCE = r"""#version 330 core
layout(location = 0) in float inCornerSelector;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec3 inCorner0;
layout(location = 3) in vec3 inCorner1;
layout(location = 4) in vec3 inCorner2;
layout(location = 5) in vec3 inCorner3;
layout(location = 6) in vec3 inCorner4;
layout(location = 7) in vec3 inCorner5;
layout(location = 8) in float inLayer;
layout(location = 9) in float inRotation;

uniform float uYaw;
uniform float uPitch;
uniform float uScale;
uniform float uAspect;
uniform float uEffectiveRatio;

out vec3 vTexCoord;
out float vDepth;
out vec2 vLocal;
flat out int vInstanceId;

vec3 rotateView(vec3 point) {
    float cy = cos(uYaw);
    float sy = sin(uYaw);
    float cp = cos(uPitch);
    float sp = sin(uPitch);
    vec3 yawed = vec3(point.x * cy + point.z * sy, point.y, -point.x * sy + point.z * cy);
    return vec3(yawed.x, yawed.y * cp - yawed.z * sp, yawed.y * sp + yawed.z * cp);
}

vec3 selectedCorner(int index) {
    if (index == 0) return inCorner0;
    if (index == 1) return inCorner1;
    if (index == 2) return inCorner2;
    if (index == 3) return inCorner3;
    if (index == 4) return inCorner4;
    return inCorner5;
}

void main() {
    vec3 world;
    if (inCornerSelector < -0.5) {
        world = normalize(inCorner0 + inCorner1 + inCorner2 + inCorner3 + inCorner4 + inCorner5);
    } else {
        world = selectedCorner(int(inCornerSelector + 0.5));
    }
    vec3 view = rotateView(world);
    gl_Position = vec4(view.x * uScale / max(uAspect, 0.0001), view.y * uScale, -view.z, 1.0);

    float angle = radians(inRotation * 60.0);
    float c = cos(angle);
    float s = sin(angle);
    vec2 centered = inUv - vec2(0.5);
    vec2 rotatedUv = vec2(centered.x * c - centered.y * s, centered.x * s + centered.y * c);
    vec2 paddedUv = vec2(0.5) + rotatedUv * uEffectiveRatio;
    vTexCoord = vec3(paddedUv, inLayer);
    vDepth = view.z;
    vLocal = centered * 2.0;
    vInstanceId = gl_InstanceID;
}
"""

FRAGMENT_SHADER_SOURCE = r"""#version 330 core
in vec3 vTexCoord;
in float vDepth;
in vec2 vLocal;
flat in int vInstanceId;
uniform sampler2DArray uTextures;
uniform int uSelectedInstance;
out vec4 outColor;

void main() {
    if (vDepth <= -0.02) {
        discard;
    }
    vec4 sampled = texture(uTextures, vTexCoord);
    if (vInstanceId == uSelectedInstance) {
        float edge = smoothstep(0.64, 0.92, length(vLocal));
        sampled.rgb = mix(sampled.rgb * 1.18, vec3(1.0, 0.83, 0.22), edge * 0.88);
    }
    outColor = sampled;
}
"""

SPHERE_VERTEX_SHADER_SOURCE = r"""#version 330 core
layout(location = 0) in vec3 inPosition;
uniform float uYaw;
uniform float uPitch;
uniform float uScale;
uniform float uAspect;
out vec3 vNormal;
out vec3 vWorld;

vec3 rotateView(vec3 point) {
    float cy = cos(uYaw);
    float sy = sin(uYaw);
    float cp = cos(uPitch);
    float sp = sin(uPitch);
    vec3 yawed = vec3(point.x * cy + point.z * sy, point.y, -point.x * sy + point.z * cy);
    return vec3(yawed.x, yawed.y * cp - yawed.z * sp, yawed.y * sp + yawed.z * cp);
}

void main() {
    vec3 world = normalize(inPosition);
    vec3 view = rotateView(world);
    vec3 inset = view * 0.9975;
    gl_Position = vec4(inset.x * uScale / max(uAspect, 0.0001), inset.y * uScale, -inset.z, 1.0);
    vNormal = normalize(view);
    vWorld = world;
}
"""

SPHERE_FRAGMENT_SHADER_SOURCE = r"""#version 330 core
in vec3 vNormal;
in vec3 vWorld;
uniform sampler2D uSurface;
uniform int uHasSurface;
out vec4 outColor;

void main() {
    vec3 world = normalize(vWorld);
    float longitude = atan(world.z, world.x);
    float latitude = asin(clamp(world.y, -1.0, 1.0));
    vec2 uv = vec2(longitude / (2.0 * 3.141592653589793) + 0.5,
                   0.5 - latitude / 3.141592653589793);
    vec3 base = uHasSurface != 0 ? texture(uSurface, uv).rgb : vec3(0.20, 0.31, 0.37);
    vec3 lightDirection = normalize(vec3(-0.48, 0.58, 0.72));
    float diffuse = 0.36 + 0.64 * max(dot(vNormal, lightDirection), 0.0);
    float rim = pow(1.0 - max(vNormal.z, 0.0), 2.2);
    vec3 color = base * diffuse + vec3(0.035, 0.045, 0.05) * rim;
    outColor = vec4(color, 1.0);
}
"""



class GpuNativeError(RuntimeError):
    pass


class GpuStreamDesyncError(GpuNativeError):
    """The incremental stream no longer matches the GPU-side instance state."""


class NativeGpuViewportHandle:
    """Thread-safe controller for a native child or top-level GPU viewport."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hwnd: int | None = None
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()

    @property
    def hwnd(self) -> int | None:
        with self._lock:
            return self._hwnd

    @property
    def thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    def _set_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            self._thread = thread

    def _set_hwnd(self, hwnd: int | None) -> None:
        with self._lock:
            self._hwnd = hwnd
        if hwnd:
            self.ready.set()

    def resize(self, width: int, height: int) -> None:
        hwnd = self.hwnd
        if os.name != "nt" or not hwnd:
            return
        user32 = ctypes.windll.user32
        user32.MoveWindow(wintypes.HWND(hwnd), 0, 0, max(1, int(width)), max(1, int(height)), True)

    def close(self, *, wait: bool = False, timeout: float = 1.0) -> None:
        hwnd = self.hwnd
        if os.name == "nt" and hwnd:
            ctypes.windll.user32.PostMessageW(wintypes.HWND(hwnd), 0x0010, 0, 0)
        thread = self.thread
        if (
            wait
            and thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(max(0.0, float(timeout)))


@dataclass(frozen=True)
class GpuPreviewLaunch:
    started: bool
    reason: str
    thread: threading.Thread | None
    viewport: NativeGpuViewportHandle | None = None


def native_gpu_supported() -> bool:
    return os.name == "nt"


def launch_gpu_preview(
    batch: GpuRenderBatch,
    yaw: float,
    pitch: float,
    zoom: float,
    *,
    topology: DualTopology | None = None,
    edit_bridge: GpuEditBridge | None = None,
    title: str = "Hex Planet GPU Editor",
    on_error: Callable[[str], None] | None = None,
    on_capabilities: Callable[[int], None] | None = None,
    streaming: bool = False,
    min_zoom: float = 0.8,
    editable: bool = True,
    surface_texture: PixelImage | None = None,
    parent_hwnd: int | None = None,
    initial_width: int = 1100,
    initial_height: int = 760,
    maximum_texture_layers: int | None = None,
    texture_memory_budget_bytes: int = DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
) -> GpuPreviewLaunch:
    if not native_gpu_supported():
        return GpuPreviewLaunch(False, "The native OpenGL preview is available on Windows only", None)

    viewport = NativeGpuViewportHandle()

    def worker() -> None:
        try:
            _Win32GpuPreview(
                batch, yaw, pitch, zoom, title,
                None if topology is None else SphericalCellPicker(topology, batch),
                edit_bridge,
                streaming=streaming, min_zoom=min_zoom, editable=editable,
                surface_texture=surface_texture, parent_hwnd=parent_hwnd,
                initial_width=initial_width, initial_height=initial_height,
                viewport_handle=viewport,
                maximum_texture_layers=maximum_texture_layers,
                texture_memory_budget_bytes=texture_memory_budget_bytes,
                on_capabilities=on_capabilities,
            ).run()
        except Exception as exc:
            logging.exception("Native GPU preview failed")
            if on_error is not None:
                try:
                    on_error(str(exc))
                except Exception:
                    logging.exception("GPU preview error callback failed")
        finally:
            viewport._set_hwnd(None)
            if edit_bridge is not None:
                edit_bridge.close()

    thread = threading.Thread(target=worker, name="hexplanet-gpu-preview", daemon=True)
    viewport._set_thread(thread)
    thread.start()
    return GpuPreviewLaunch(True, "GPU preview thread started", thread, viewport)


if os.name == "nt":
    LRESULT = ctypes.c_ssize_t
    WPARAM = ctypes.c_size_t
    LPARAM = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", ctypes.c_ubyte),
            ("cColorBits", ctypes.c_ubyte),
            ("cRedBits", ctypes.c_ubyte),
            ("cRedShift", ctypes.c_ubyte),
            ("cGreenBits", ctypes.c_ubyte),
            ("cGreenShift", ctypes.c_ubyte),
            ("cBlueBits", ctypes.c_ubyte),
            ("cBlueShift", ctypes.c_ubyte),
            ("cAlphaBits", ctypes.c_ubyte),
            ("cAlphaShift", ctypes.c_ubyte),
            ("cAccumBits", ctypes.c_ubyte),
            ("cAccumRedBits", ctypes.c_ubyte),
            ("cAccumGreenBits", ctypes.c_ubyte),
            ("cAccumBlueBits", ctypes.c_ubyte),
            ("cAccumAlphaBits", ctypes.c_ubyte),
            ("cDepthBits", ctypes.c_ubyte),
            ("cStencilBits", ctypes.c_ubyte),
            ("cAuxBuffers", ctypes.c_ubyte),
            ("iLayerType", ctypes.c_ubyte),
            ("bReserved", ctypes.c_ubyte),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    class MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("message", wintypes.UINT),
            ("wParam", WPARAM),
            ("lParam", LPARAM),
            ("time", wintypes.DWORD),
            ("pt", wintypes.POINT),
        ]
else:
    WNDPROC = object  # type: ignore[assignment,misc]


class _Win32GpuPreview:
    _registry: dict[int, "_Win32GpuPreview"] = {}
    _class_atom: int = 0
    _wndproc_ref = None

    WM_DESTROY = 0x0002
    WM_SIZE = 0x0005
    WM_CLOSE = 0x0010
    WM_KEYDOWN = 0x0100
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    WM_MOUSEMOVE = 0x0200
    WM_MOUSEWHEEL = 0x020A
    VK_ESCAPE = 0x1B
    VK_CONTROL = 0x11
    VK_E = 0x45
    VK_P = 0x50
    VK_R = 0x52
    VK_S = 0x53
    PM_REMOVE = 0x0001
    WS_OVERLAPPEDWINDOW = 0x00CF0000
    WS_CHILD = 0x40000000
    WS_CLIPSIBLINGS = 0x04000000
    WS_CLIPCHILDREN = 0x02000000
    WS_VISIBLE = 0x10000000
    CS_OWNDC = 0x0020
    IDC_ARROW = 32512
    PFD_DRAW_TO_WINDOW = 0x00000004
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_DOUBLEBUFFER = 0x00000001
    PFD_TYPE_RGBA = 0
    PFD_MAIN_PLANE = 0

    GL_COLOR_BUFFER_BIT = 0x00004000
    GL_DEPTH_BUFFER_BIT = 0x00000100
    GL_DEPTH_TEST = 0x0B71
    GL_BLEND = 0x0BE2
    GL_SRC_ALPHA = 0x0302
    GL_ONE_MINUS_SRC_ALPHA = 0x0303
    GL_LESS = 0x0201
    GL_FLOAT = 0x1406
    GL_FALSE = 0
    GL_TRIANGLES = 0x0004
    GL_ARRAY_BUFFER = 0x8892
    GL_STATIC_DRAW = 0x88E4
    GL_DYNAMIC_DRAW = 0x88E8
    GL_TEXTURE0 = 0x84C0
    GL_TEXTURE1 = 0x84C1
    GL_TEXTURE_2D = 0x0DE1
    GL_TEXTURE_2D_ARRAY = 0x8C1A
    GL_TEXTURE_MIN_FILTER = 0x2801
    GL_TEXTURE_MAG_FILTER = 0x2800
    GL_TEXTURE_WRAP_S = 0x2802
    GL_TEXTURE_WRAP_T = 0x2803
    GL_TEXTURE_WRAP_R = 0x8072
    GL_NEAREST = 0x2600
    GL_LINEAR = 0x2601
    GL_REPEAT = 0x2901
    GL_CLAMP_TO_EDGE = 0x812F
    GL_RGBA8 = 0x8058
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_UNPACK_ALIGNMENT = 0x0CF5
    GL_VERTEX_SHADER = 0x8B31
    GL_FRAGMENT_SHADER = 0x8B30
    GL_COMPILE_STATUS = 0x8B81
    GL_LINK_STATUS = 0x8B82
    GL_INFO_LOG_LENGTH = 0x8B84
    GL_VERSION = 0x1F02
    GL_MAX_ARRAY_TEXTURE_LAYERS = 0x88FF

    def __init__(
        self,
        batch: GpuRenderBatch,
        yaw: float,
        pitch: float,
        zoom: float,
        title: str,
        picker: SphericalCellPicker | None,
        edit_bridge: GpuEditBridge | None,
        *,
        streaming: bool = False,
        min_zoom: float = 0.8,
        editable: bool = True,
        surface_texture: PixelImage | None = None,
        parent_hwnd: int | None = None,
        initial_width: int = 1100,
        initial_height: int = 760,
        viewport_handle: NativeGpuViewportHandle | None = None,
        maximum_texture_layers: int | None = None,
        texture_memory_budget_bytes: int = DEFAULT_GPU_TEXTURE_BUDGET_BYTES,
        on_capabilities: Callable[[int], None] | None = None,
    ) -> None:
        if os.name != "nt":
            raise GpuNativeError("The native GPU preview requires Windows")
        self.batch = batch
        self.instances = list(batch.instances)
        self.instance_by_cell = {instance.cell_id: index for index, instance in enumerate(batch.instances)}
        self.picker = picker
        self.edit_bridge = edit_bridge
        self.streaming = bool(streaming)
        self.editable = bool(editable and picker is not None)
        self.min_zoom = max(0.25, min(512.0, float(min_zoom)))
        self.base_title = title
        self.selected_instance = -1
        self.last_status = t("右键拖动旋转；按住左键连续绘制；P/E 工具；Ctrl+S 保存")
        self.texture_pixels: list[bytes | None] = []
        self.texture_sources: list[Path | None] = []
        self.texture_placeholder = b""
        self.texture_key_to_layer: dict[str, int] = {}
        self.texture_capacity = 0
        self.maximum_texture_layers = 0
        self.driver_maximum_texture_layers = 0
        self.requested_maximum_texture_layers = (
            None if maximum_texture_layers is None else max(2, int(maximum_texture_layers))
        )
        self.texture_memory_budget_bytes = max(1, int(texture_memory_budget_bytes))
        self.on_capabilities = on_capabilities
        self.yaw = yaw
        self.pitch = pitch
        self.zoom = max(self.min_zoom, min(512.0, zoom))
        self.title = title
        self.width = max(1, int(initial_width))
        self.height = max(1, int(initial_height))
        self.parent_hwnd = None if parent_hwnd is None else int(parent_hwnd)
        self.embedded = self.parent_hwnd is not None
        self.viewport_handle = viewport_handle
        self.surface_image = surface_texture
        self.hwnd = None
        self.hdc = None
        self.context = None
        self.running = True
        self.dragging = False
        self.painting = False
        self.paint_stroke_id = 0
        self.last_paint_cell: int | None = None
        self.last_mouse = (0, 0)
        self.view_dirty = self.streaming
        self.last_view_submit = 0.0
        self.interactive_until = 0.0
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.kernel32 = ctypes.windll.kernel32
        self.opengl32 = ctypes.windll.opengl32
        self._configure_win32_signatures()
        self.gl: dict[str, object] = {}
        self.program = 0
        self.vao = ctypes.c_uint(0)
        self.buffers = (ctypes.c_uint * 2)()
        self.sphere_program = 0
        self.sphere_vao = ctypes.c_uint(0)
        self.sphere_buffer = ctypes.c_uint(0)
        self.sphere_vertex_count = 0
        self.texture = ctypes.c_uint(0)
        self.surface_texture = ctypes.c_uint(0)
        self.instance_capacity = 0
        self.pending_texture_uploads: dict[int, bytes] = {}
        self.last_upload_title_update = 0.0
        self.last_submitted_view: tuple[float, float, float, int, int] | None = None
        self.force_view_submit = self.streaming
        self.stream_resync_pending = False
        self.frame_dirty = True
        self.last_render_time = 0.0

    def _configure_win32_signatures(self) -> None:
        self.user32.DefWindowProcW.restype = ctypes.c_ssize_t
        self.user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, WPARAM, LPARAM]
        self.user32.RegisterClassW.restype = wintypes.WORD
        self.user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        self.user32.CreateWindowExW.restype = wintypes.HWND
        self.user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
        ]
        self.user32.GetDC.restype = wintypes.HDC
        self.user32.GetDC.argtypes = [wintypes.HWND]
        self.user32.ReleaseDC.restype = ctypes.c_int
        self.user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self.user32.DestroyWindow.restype = wintypes.BOOL
        self.user32.DestroyWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.PeekMessageW.restype = wintypes.BOOL
        self.user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        self.user32.TranslateMessage.restype = wintypes.BOOL
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
        self.user32.PostQuitMessage.argtypes = [ctypes.c_int]
        self.user32.PostMessageW.restype = wintypes.BOOL
        self.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, WPARAM, LPARAM]
        self.user32.LoadCursorW.restype = wintypes.HANDLE
        self.user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
        self.user32.SetCapture.restype = wintypes.HWND
        self.user32.SetCapture.argtypes = [wintypes.HWND]
        self.user32.SetFocus.restype = wintypes.HWND
        self.user32.SetFocus.argtypes = [wintypes.HWND]
        self.user32.ReleaseCapture.restype = wintypes.BOOL
        self.user32.ReleaseCapture.argtypes = []
        self.user32.SetWindowTextW.restype = wintypes.BOOL
        self.user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        self.user32.MoveWindow.restype = wintypes.BOOL
        self.user32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
        self.user32.GetKeyState.restype = ctypes.c_short
        self.user32.GetKeyState.argtypes = [ctypes.c_int]

        self.gdi32.ChoosePixelFormat.restype = ctypes.c_int
        self.gdi32.ChoosePixelFormat.argtypes = [wintypes.HDC, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
        self.gdi32.SetPixelFormat.restype = wintypes.BOOL
        self.gdi32.SetPixelFormat.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.POINTER(PIXELFORMATDESCRIPTOR)]
        self.gdi32.SwapBuffers.restype = wintypes.BOOL
        self.gdi32.SwapBuffers.argtypes = [wintypes.HDC]

        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetLastError.restype = wintypes.DWORD
        self.kernel32.GetLastError.argtypes = []

        self.opengl32.wglCreateContext.restype = wintypes.HANDLE
        self.opengl32.wglCreateContext.argtypes = [wintypes.HDC]
        self.opengl32.wglMakeCurrent.restype = wintypes.BOOL
        self.opengl32.wglMakeCurrent.argtypes = [wintypes.HDC, wintypes.HANDLE]
        self.opengl32.wglDeleteContext.restype = wintypes.BOOL
        self.opengl32.wglDeleteContext.argtypes = [wintypes.HANDLE]
        self.opengl32.wglGetProcAddress.restype = ctypes.c_void_p
        self.opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        self.opengl32.glGetString.restype = ctypes.c_char_p
        self.opengl32.glGetString.argtypes = [ctypes.c_uint]
        self.opengl32.glGetIntegerv.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
        self.opengl32.glViewport.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.opengl32.glClearColor.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.opengl32.glClear.argtypes = [ctypes.c_uint]
        self.opengl32.glEnable.argtypes = [ctypes.c_uint]
        self.opengl32.glDepthFunc.argtypes = [ctypes.c_uint]
        self.opengl32.glBlendFunc.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.opengl32.glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        self.opengl32.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.opengl32.glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
        self.opengl32.glPixelStorei.argtypes = [ctypes.c_uint, ctypes.c_int]
        self.opengl32.glDeleteTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]

    @classmethod
    def _register_class(cls, user32, kernel32) -> None:
        if cls._class_atom:
            return

        @WNDPROC
        def wndproc(hwnd, message, wparam, lparam):
            instance = cls._registry.get(int(hwnd))
            if instance is not None:
                return instance._window_proc(hwnd, message, wparam, lparam)
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        cls._wndproc_ref = wndproc
        class_name = "HexPlanetGpuPreviewWindow"
        window_class = WNDCLASSW()
        window_class.style = cls.CS_OWNDC
        window_class.lpfnWndProc = wndproc
        window_class.hInstance = kernel32.GetModuleHandleW(None)
        window_class.hCursor = user32.LoadCursorW(None, ctypes.c_void_p(cls.IDC_ARROW))
        window_class.lpszClassName = class_name
        atom = user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            error = kernel32.GetLastError()
            if error != 1410:
                raise GpuNativeError(f"RegisterClassW failed with error {error}")
            atom = 1
        cls._class_atom = int(atom)

    def run(self) -> None:
        self._register_class(self.user32, self.kernel32)
        instance = self.kernel32.GetModuleHandleW(None)
        style = (
            self.WS_CHILD | self.WS_VISIBLE | self.WS_CLIPSIBLINGS | self.WS_CLIPCHILDREN
            if self.embedded
            else self.WS_OVERLAPPEDWINDOW | self.WS_VISIBLE
        )
        self.hwnd = self.user32.CreateWindowExW(
            0, "HexPlanetGpuPreviewWindow", self.title, style,
            0 if self.embedded else 100, 0 if self.embedded else 100,
            self.width, self.height,
            None if self.parent_hwnd is None else wintypes.HWND(self.parent_hwnd),
            None, instance, None,
        )
        if not self.hwnd:
            raise GpuNativeError(f"CreateWindowExW failed with error {self.kernel32.GetLastError()}")
        self._registry[int(self.hwnd)] = self
        if self.viewport_handle is not None:
            self.viewport_handle._set_hwnd(int(self.hwnd))
        try:
            self._create_context()
            self._create_resources()
            if self.on_capabilities is not None:
                self.on_capabilities(self.maximum_texture_layers)
            self._message_loop()
        finally:
            self._cleanup()
            if self.hwnd:
                self._registry.pop(int(self.hwnd), None)

    def _create_context(self) -> None:
        self.hdc = self.user32.GetDC(self.hwnd)
        if not self.hdc:
            raise GpuNativeError("GetDC failed")
        pfd = PIXELFORMATDESCRIPTOR()
        pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
        pfd.nVersion = 1
        pfd.dwFlags = self.PFD_DRAW_TO_WINDOW | self.PFD_SUPPORT_OPENGL | self.PFD_DOUBLEBUFFER
        pfd.iPixelType = self.PFD_TYPE_RGBA
        pfd.cColorBits = 32
        pfd.cAlphaBits = 8
        pfd.cDepthBits = 24
        pfd.cStencilBits = 8
        pfd.iLayerType = self.PFD_MAIN_PLANE
        pixel_format = self.gdi32.ChoosePixelFormat(self.hdc, ctypes.byref(pfd))
        if not pixel_format:
            raise GpuNativeError("ChoosePixelFormat failed")
        if not self.gdi32.SetPixelFormat(self.hdc, pixel_format, ctypes.byref(pfd)):
            raise GpuNativeError("SetPixelFormat failed")

        temporary = self.opengl32.wglCreateContext(self.hdc)
        if not temporary or not self.opengl32.wglMakeCurrent(self.hdc, temporary):
            raise GpuNativeError("Cannot create the temporary OpenGL context")
        create_attribs = self._load_proc(
            "wglCreateContextAttribsARB",
            wintypes.HANDLE,
            [wintypes.HDC, wintypes.HANDLE, ctypes.POINTER(ctypes.c_int)],
            required=False,
        )
        context = temporary
        if create_attribs is not None:
            attributes = (ctypes.c_int * 9)(
                0x2091,
                3,
                0x2092,
                3,
                0x9126,
                0x00000001,
                0,
                0,
                0,
            )
            modern = create_attribs(self.hdc, None, attributes)
            if modern:
                self.opengl32.wglMakeCurrent(None, None)
                self.opengl32.wglDeleteContext(temporary)
                context = modern
                if not self.opengl32.wglMakeCurrent(self.hdc, context):
                    raise GpuNativeError("Cannot activate the OpenGL 3.3 context")
        self.context = context
        version_raw = self.opengl32.glGetString(self.GL_VERSION)
        version = "unknown" if not version_raw else version_raw.decode("ascii", errors="replace")
        major, minor = _parse_gl_version(version)
        if (major, minor) < (3, 3):
            raise GpuNativeError(f"OpenGL 3.3 or newer is required; driver reported {version}")
        self._load_gl_functions()
        swap_interval = self._load_proc(
            "wglSwapIntervalEXT",
            ctypes.c_int,
            [ctypes.c_int],
            required=False,
        )
        if swap_interval is not None:
            swap_interval(1)

    def _load_proc(self, name, restype, argtypes, *, required=True):
        address = self.opengl32.wglGetProcAddress(name.encode("ascii"))
        if address in (None, 0, 1, 2, 3, ctypes.c_void_p(-1).value):
            try:
                function = getattr(self.opengl32, name)
            except AttributeError:
                if required:
                    raise GpuNativeError(f"OpenGL function is unavailable: {name}")
                return None
            function.restype = restype
            function.argtypes = argtypes
            return function
        prototype = ctypes.WINFUNCTYPE(restype, *argtypes)
        return prototype(address)

    def _load_gl_functions(self) -> None:
        u = ctypes.c_uint
        i = ctypes.c_int
        f = ctypes.c_float
        size = ctypes.c_ssize_t
        void_p = ctypes.c_void_p
        self.gl = {
            "glCreateShader": self._load_proc("glCreateShader", u, [u]),
            "glShaderSource": self._load_proc("glShaderSource", None, [u, i, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(i)]),
            "glCompileShader": self._load_proc("glCompileShader", None, [u]),
            "glGetShaderiv": self._load_proc("glGetShaderiv", None, [u, u, ctypes.POINTER(i)]),
            "glGetShaderInfoLog": self._load_proc("glGetShaderInfoLog", None, [u, i, ctypes.POINTER(i), ctypes.c_char_p]),
            "glDeleteShader": self._load_proc("glDeleteShader", None, [u]),
            "glCreateProgram": self._load_proc("glCreateProgram", u, []),
            "glAttachShader": self._load_proc("glAttachShader", None, [u, u]),
            "glLinkProgram": self._load_proc("glLinkProgram", None, [u]),
            "glGetProgramiv": self._load_proc("glGetProgramiv", None, [u, u, ctypes.POINTER(i)]),
            "glGetProgramInfoLog": self._load_proc("glGetProgramInfoLog", None, [u, i, ctypes.POINTER(i), ctypes.c_char_p]),
            "glUseProgram": self._load_proc("glUseProgram", None, [u]),
            "glDeleteProgram": self._load_proc("glDeleteProgram", None, [u]),
            "glGenVertexArrays": self._load_proc("glGenVertexArrays", None, [i, ctypes.POINTER(u)]),
            "glBindVertexArray": self._load_proc("glBindVertexArray", None, [u]),
            "glDeleteVertexArrays": self._load_proc("glDeleteVertexArrays", None, [i, ctypes.POINTER(u)]),
            "glGenBuffers": self._load_proc("glGenBuffers", None, [i, ctypes.POINTER(u)]),
            "glBindBuffer": self._load_proc("glBindBuffer", None, [u, u]),
            "glBufferData": self._load_proc("glBufferData", None, [u, size, void_p, u]),
            "glBufferSubData": self._load_proc("glBufferSubData", None, [u, size, size, void_p]),
            "glDeleteBuffers": self._load_proc("glDeleteBuffers", None, [i, ctypes.POINTER(u)]),
            "glEnableVertexAttribArray": self._load_proc("glEnableVertexAttribArray", None, [u]),
            "glVertexAttribPointer": self._load_proc("glVertexAttribPointer", None, [u, i, u, ctypes.c_ubyte, i, void_p]),
            "glVertexAttribDivisor": self._load_proc("glVertexAttribDivisor", None, [u, u]),
            "glGetUniformLocation": self._load_proc("glGetUniformLocation", i, [u, ctypes.c_char_p]),
            "glUniform1i": self._load_proc("glUniform1i", None, [i, i]),
            "glUniform1f": self._load_proc("glUniform1f", None, [i, f]),
            "glActiveTexture": self._load_proc("glActiveTexture", None, [u]),
            "glTexImage2D": self._load_proc("glTexImage2D", None, [u, i, i, i, i, i, u, u, void_p]),
            "glTexSubImage2D": self._load_proc("glTexSubImage2D", None, [u, i, i, i, i, i, u, u, void_p]),
            "glTexImage3D": self._load_proc("glTexImage3D", None, [u, i, i, i, i, i, i, u, u, void_p]),
            "glTexSubImage3D": self._load_proc("glTexSubImage3D", None, [u, i, i, i, i, i, i, i, u, u, void_p]),
            "glDrawArraysInstanced": self._load_proc("glDrawArraysInstanced", None, [u, i, i, i]),
        }

    def _create_resources(self) -> None:
        side = self.batch.padded_size
        layer_count = len(self.batch.texture_layers)
        maximum_layers = ctypes.c_int(0)
        self.opengl32.glGetIntegerv(self.GL_MAX_ARRAY_TEXTURE_LAYERS, ctypes.byref(maximum_layers))
        self.driver_maximum_texture_layers = maximum_layers.value
        self._configure_texture_layer_limit(side)
        if layer_count > self.maximum_texture_layers:
            raise GpuNativeError(
                f"Texture array needs {layer_count} layers, "
                f"but the GPU supports {self.maximum_texture_layers}"
            )

        vertex_shader = self._compile_shader(self.GL_VERTEX_SHADER, VERTEX_SHADER_SOURCE)
        fragment_shader = self._compile_shader(self.GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE)
        create_program = self.gl["glCreateProgram"]
        self.program = create_program()
        self.gl["glAttachShader"](self.program, vertex_shader)
        self.gl["glAttachShader"](self.program, fragment_shader)
        self.gl["glLinkProgram"](self.program)
        status = ctypes.c_int(0)
        self.gl["glGetProgramiv"](self.program, self.GL_LINK_STATUS, ctypes.byref(status))
        self.gl["glDeleteShader"](vertex_shader)
        self.gl["glDeleteShader"](fragment_shader)
        if not status.value:
            raise GpuNativeError(f"OpenGL program link failed: {self._program_log(self.program)}")

        sphere_vertex_shader = self._compile_shader(self.GL_VERTEX_SHADER, SPHERE_VERTEX_SHADER_SOURCE)
        sphere_fragment_shader = self._compile_shader(self.GL_FRAGMENT_SHADER, SPHERE_FRAGMENT_SHADER_SOURCE)
        self.sphere_program = create_program()
        self.gl["glAttachShader"](self.sphere_program, sphere_vertex_shader)
        self.gl["glAttachShader"](self.sphere_program, sphere_fragment_shader)
        self.gl["glLinkProgram"](self.sphere_program)
        sphere_status = ctypes.c_int(0)
        self.gl["glGetProgramiv"](self.sphere_program, self.GL_LINK_STATUS, ctypes.byref(sphere_status))
        self.gl["glDeleteShader"](sphere_vertex_shader)
        self.gl["glDeleteShader"](sphere_fragment_shader)
        if not sphere_status.value:
            raise GpuNativeError(
                f"OpenGL sphere program link failed: {self._program_log(self.sphere_program)}"
            )

        sphere_mesh = _unit_sphere_mesh_bytes()
        self.sphere_vertex_count = len(sphere_mesh) // 12
        sphere_mesh_buffer = ctypes.create_string_buffer(sphere_mesh)
        self.gl["glGenVertexArrays"](1, ctypes.byref(self.sphere_vao))
        self.gl["glBindVertexArray"](self.sphere_vao.value)
        self.gl["glGenBuffers"](1, ctypes.byref(self.sphere_buffer))
        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.sphere_buffer.value)
        self.gl["glBufferData"](
            self.GL_ARRAY_BUFFER,
            len(sphere_mesh),
            ctypes.cast(sphere_mesh_buffer, ctypes.c_void_p),
            self.GL_STATIC_DRAW,
        )
        self._attribute(0, 3, 12, 0, 0)

        self.opengl32.glGenTextures(1, ctypes.byref(self.surface_texture))
        self._upload_surface_texture(self.surface_image)

        self.gl["glGenVertexArrays"](1, ctypes.byref(self.vao))
        self.gl["glBindVertexArray"](self.vao.value)
        self.gl["glGenBuffers"](2, self.buffers)

        mesh = unit_hex_mesh_bytes()
        mesh_buffer = ctypes.create_string_buffer(mesh)
        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[0])
        self.gl["glBufferData"](
            self.GL_ARRAY_BUFFER,
            len(mesh),
            ctypes.cast(mesh_buffer, ctypes.c_void_p),
            self.GL_STATIC_DRAW,
        )
        self._attribute(0, 1, 12, 0, 0)
        self._attribute(1, 2, 12, 4, 0)

        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[1])
        self.instance_capacity = max(1, _next_power_of_two(max(1, len(self.instances))))
        self.gl["glBufferData"](
            self.GL_ARRAY_BUFFER,
            self.instance_capacity * INSTANCE_STRUCT.size,
            None,
            self.GL_DYNAMIC_DRAW,
        )
        if self.instances:
            self._upload_instance_slots(range(len(self.instances)))
        stride = INSTANCE_STRUCT.size
        for location in range(2, 8):
            self._attribute(location, 3, stride, (location - 2) * 12, 1)
        self._attribute(8, 1, stride, 72, 1)
        self._attribute(9, 1, stride, 76, 1)

        self.texture_placeholder = bytes((63, 73, 82, 255)) * (side * side)
        self.texture_pixels = []
        self.texture_sources = []
        for layer in self.batch.texture_layers:
            if layer.kind == "brush" and layer.source_path is not None:
                self.texture_pixels.append(None)
                self.texture_sources.append(Path(layer.source_path))
            else:
                self.texture_pixels.append(build_texture_layer_payload(layer, side))
                self.texture_sources.append(None)
        self.texture_key_to_layer = {
            layer.key: index for index, layer in enumerate(self.batch.texture_layers)
        }

        self.opengl32.glGenTextures(1, ctypes.byref(self.texture))
        self.gl["glActiveTexture"](self.GL_TEXTURE0)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D_ARRAY, self.texture.value)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D_ARRAY, self.GL_TEXTURE_MIN_FILTER, self.GL_NEAREST)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D_ARRAY, self.GL_TEXTURE_MAG_FILTER, self.GL_NEAREST)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D_ARRAY, self.GL_TEXTURE_WRAP_S, self.GL_CLAMP_TO_EDGE)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D_ARRAY, self.GL_TEXTURE_WRAP_T, self.GL_CLAMP_TO_EDGE)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D_ARRAY, self.GL_TEXTURE_WRAP_R, self.GL_CLAMP_TO_EDGE)
        self.opengl32.glPixelStorei(self.GL_UNPACK_ALIGNMENT, 1)
        self.texture_capacity = next_texture_capacity(
            layer_count,
            0,
            self.maximum_texture_layers,
        )
        self._allocate_texture_storage()

        self.gl["glUseProgram"](self.program)
        self.gl["glUniform1i"](self._uniform("uTextures"), 0)
        self.gl["glUniform1f"](
            self._uniform("uEffectiveRatio"),
            self.batch.effective_size / self.batch.padded_size,
        )
        self.gl["glUniform1i"](self._uniform("uSelectedInstance"), -1)
        self.gl["glUseProgram"](self.sphere_program)
        self.gl["glUniform1i"](self._uniform_for(self.sphere_program, "uSurface"), 1)
        self.gl["glUniform1i"](
            self._uniform_for(self.sphere_program, "uHasSurface"),
            1 if self.surface_image is not None else 0,
        )
        self.opengl32.glEnable(self.GL_DEPTH_TEST)
        self.opengl32.glDepthFunc(self.GL_LESS)
        self.opengl32.glEnable(self.GL_BLEND)
        self.opengl32.glBlendFunc(self.GL_SRC_ALPHA, self.GL_ONE_MINUS_SRC_ALPHA)
        self._update_window_title()

    def _upload_surface_texture(self, image: PixelImage | None) -> None:
        self.gl["glActiveTexture"](self.GL_TEXTURE1)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D, self.surface_texture.value)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_MIN_FILTER, self.GL_LINEAR)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_MAG_FILTER, self.GL_LINEAR)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_WRAP_S, self.GL_REPEAT)
        self.opengl32.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_WRAP_T, self.GL_CLAMP_TO_EDGE)
        self.opengl32.glPixelStorei(self.GL_UNPACK_ALIGNMENT, 1)
        if image is None:
            width = height = 1
            rgba = bytes((51, 79, 94, 255))
        else:
            width = int(image.width)
            height = int(image.height)
            rgba = _pixel_image_rgba(image)
        buffer = ctypes.create_string_buffer(rgba)
        self.gl["glTexImage2D"](
            self.GL_TEXTURE_2D, 0, self.GL_RGBA8, width, height, 0,
            self.GL_RGBA, self.GL_UNSIGNED_BYTE, ctypes.cast(buffer, ctypes.c_void_p),
        )
        self.surface_image = image
        if self.sphere_program:
            self.gl["glUseProgram"](self.sphere_program)
            self.gl["glUniform1i"](
                self._uniform_for(self.sphere_program, "uHasSurface"),
                1 if image is not None else 0,
            )

    def _allocate_texture_storage(self) -> None:
        side = self.batch.padded_size
        self.gl["glActiveTexture"](self.GL_TEXTURE0)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D_ARRAY, self.texture.value)
        self.gl["glTexImage3D"](
            self.GL_TEXTURE_2D_ARRAY,
            0,
            self.GL_RGBA8,
            side,
            side,
            self.texture_capacity,
            0,
            self.GL_RGBA,
            self.GL_UNSIGNED_BYTE,
            None,
        )
        for layer in range(len(self.texture_pixels)):
            self._upload_texture_layer(layer, self._texture_layer_pixels(layer))
            if self.texture_sources[layer] is not None:
                self.texture_pixels[layer] = None
        self.pending_texture_uploads.clear()

    def _texture_layer_pixels(self, layer: int) -> bytes:
        pixels = self.texture_pixels[layer]
        if pixels is not None:
            return pixels
        source = self.texture_sources[layer] if layer < len(self.texture_sources) else None
        if source is None:
            return self.texture_placeholder
        image = read_png_pixels(source)
        if image.width != self.batch.padded_size or image.height != self.batch.padded_size:
            raise GpuNativeError(
                f"Texture layer size mismatch for {source}: expected "
                f"{self.batch.padded_size}x{self.batch.padded_size}, "
                f"received {image.width}x{image.height}"
            )
        return _pixel_image_rgba(image)

    def _configure_texture_layer_limit(self, padded_size: int) -> None:
        configured = self.driver_maximum_texture_layers
        if self.requested_maximum_texture_layers is not None:
            configured = min(configured, self.requested_maximum_texture_layers)
        self.maximum_texture_layers = budgeted_texture_layer_limit(
            padded_size,
            budget_bytes=self.texture_memory_budget_bytes,
            configured_limit=configured,
        )

    def _upload_texture_layer(self, layer: int, pixels: bytes) -> None:
        self.gl["glActiveTexture"](self.GL_TEXTURE0)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D_ARRAY, self.texture.value)
        expected = self.batch.padded_size * self.batch.padded_size * 4
        if len(pixels) != expected:
            raise GpuNativeError(
                f"Texture update byte count mismatch: expected {expected}, received {len(pixels)}"
            )
        buffer = ctypes.create_string_buffer(pixels)
        self.gl["glTexSubImage3D"](
            self.GL_TEXTURE_2D_ARRAY,
            0,
            0,
            0,
            layer,
            self.batch.padded_size,
            self.batch.padded_size,
            1,
            self.GL_RGBA,
            self.GL_UNSIGNED_BYTE,
            ctypes.cast(buffer, ctypes.c_void_p),
        )

    def _ensure_texture_layer(self, patch: GpuCellPatch) -> int:
        if not hasattr(self, "texture_sources"):
            self.texture_sources = [None] * len(self.texture_pixels)
        existing = self.texture_key_to_layer.get(patch.texture_key)
        if existing is not None:
            return existing
        if patch.pixels_rgba is None:
            raise GpuNativeError(f"Texture payload is missing for key {patch.texture_key}")
        if patch.texture_width != self.batch.padded_size or patch.texture_height != self.batch.padded_size:
            raise GpuNativeError(
                f"Texture update size mismatch: expected {self.batch.padded_size}x{self.batch.padded_size}, "
                f"received {patch.texture_width}x{patch.texture_height}"
            )
        if len(self.texture_pixels) >= self.maximum_texture_layers:
            raise GpuNativeError(
                f"GPU texture-array layer limit reached ({self.maximum_texture_layers})"
            )
        if len(self.texture_pixels) >= self.texture_capacity:
            self.texture_capacity = next_texture_capacity(
                len(self.texture_pixels) + 1,
                self.texture_capacity,
                self.maximum_texture_layers,
            )
            self._allocate_texture_storage()
        layer = len(self.texture_pixels)
        self.texture_pixels.append(patch.pixels_rgba)
        self.texture_sources.append(None)
        self.texture_key_to_layer[patch.texture_key] = layer
        self._upload_texture_layer(layer, patch.pixels_rgba)
        return layer

    def _apply_cell_patch(self, patch: GpuCellPatch) -> None:
        instance_index = self.instance_by_cell.get(patch.cell_id)
        if instance_index is None:
            self.last_status = t("CellId {cell_id} 不在当前 GPU 实例批次中", cell_id=patch.cell_id)
            self._update_window_title()
            return
        layer = self._ensure_texture_layer(patch)
        updated = replace(
            self.instances[instance_index],
            texture_layer=layer,
            rotation=patch.rotation,
        )
        self.instances[instance_index] = updated
        payload = updated.packed()
        buffer = ctypes.create_string_buffer(payload)
        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[1])
        self.gl["glBufferSubData"](
            self.GL_ARRAY_BUFFER,
            instance_index * INSTANCE_STRUCT.size,
            len(payload),
            ctypes.cast(buffer, ctypes.c_void_p),
        )
        self.selected_instance = instance_index
        self.last_status = patch.message
        self._update_window_title()

    def _apply_surface_texture_patch(self, patch: GpuSurfaceTexturePatch) -> None:
        if patch.width <= 0 or patch.height <= 0:
            raise GpuNativeError("Surface texture dimensions must be positive")
        if patch.channels not in (3, 4):
            raise GpuNativeError("Surface texture must be RGB or RGBA")
        expected = patch.width * patch.height * patch.channels
        if len(patch.pixels) != expected:
            raise GpuNativeError(
                f"Surface texture byte count mismatch: expected {expected}, received {len(patch.pixels)}"
            )
        image = PixelImage(patch.width, patch.height, patch.channels, patch.pixels)
        self._upload_surface_texture(image)
        self.last_status = patch.message
        self._update_window_title()

    def _apply_surface_region_patch(self, patch: GpuSurfaceRegionPatch) -> None:
        image = self.surface_image
        if image is None:
            raise GpuNativeError("Cannot update a surface region before the surface exists")
        if patch.width <= 0 or patch.height <= 0:
            raise GpuNativeError("Surface region dimensions must be positive")
        if patch.x < 0 or patch.y < 0:
            raise GpuNativeError("Surface region origin cannot be negative")
        if patch.x + patch.width > image.width or patch.y + patch.height > image.height:
            raise GpuNativeError("Surface region is outside the current texture")
        if patch.channels not in (3, 4):
            raise GpuNativeError("Surface region must be RGB or RGBA")
        expected = patch.width * patch.height * patch.channels
        if len(patch.pixels) != expected:
            raise GpuNativeError(
                f"Surface region byte count mismatch: expected {expected}, "
                f"received {len(patch.pixels)}"
            )
        rgba = _pixel_image_rgba(
            PixelImage(patch.width, patch.height, patch.channels, patch.pixels)
        )
        buffer = ctypes.create_string_buffer(rgba)
        self.gl["glActiveTexture"](self.GL_TEXTURE1)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D, self.surface_texture.value)
        self.opengl32.glPixelStorei(self.GL_UNPACK_ALIGNMENT, 1)
        self.gl["glTexSubImage2D"](
            self.GL_TEXTURE_2D,
            0,
            patch.x,
            patch.y,
            patch.width,
            patch.height,
            self.GL_RGBA,
            self.GL_UNSIGNED_BYTE,
            ctypes.cast(buffer, ctypes.c_void_p),
        )
        self.last_status = patch.message
        self._update_window_title()

    def _flush_pending_texture_uploads(self) -> None:
        if not self.pending_texture_uploads:
            return
        side = self.batch.padded_size
        layer_bytes = max(1, side * side * 4)
        byte_budget = (8 if self.dragging else 48) * 1024 * 1024
        layer_cap = 128 if self.dragging else 512
        budget = max(2, min(layer_cap, byte_budget // layer_bytes))
        deadline = time.perf_counter() + (0.004 if self.dragging else 0.012)
        uploaded = 0
        for layer in sorted(tuple(self.pending_texture_uploads))[:budget]:
            pixels = self.pending_texture_uploads.pop(layer)
            self._upload_texture_layer(layer, pixels)
            if layer < len(self.texture_sources) and self.texture_sources[layer] is not None:
                self.texture_pixels[layer] = None
            self.frame_dirty = True
            uploaded += 1
            if uploaded >= 2 and time.perf_counter() >= deadline:
                break
        now = time.monotonic()
        if not self.pending_texture_uploads or now - self.last_upload_title_update >= 0.2:
            self.last_upload_title_update = now
            self._update_window_title()

    def _poll_edit_bridge(self) -> None:
        bridge = self.edit_bridge
        if bridge is None:
            return
        if bridge.closed:
            self.running = False
            return
        for patch in bridge.poll_patches():
            logging.debug(
                "GPU applying %s request=%s instances=%s current=%s",
                type(patch).__name__,
                getattr(patch, "request_id", None),
                (
                    getattr(patch, "instance_count", None)
                    if not isinstance(patch, GpuBatchResetPatch)
                    else patch.batch.instance_count
                ),
                len(self.instances),
            )
            self.frame_dirty = True
            if isinstance(patch, GpuCellPatch):
                self._apply_cell_patch(patch)
            elif isinstance(patch, GpuSurfaceTexturePatch):
                self._apply_surface_texture_patch(patch)
            elif isinstance(patch, GpuSurfaceRegionPatch):
                self._apply_surface_region_patch(patch)
            elif isinstance(patch, GpuStreamPatch):
                if self.stream_resync_pending:
                    continue
                try:
                    self._apply_stream_patch(patch)
                except GpuStreamDesyncError as exc:
                    logging.exception(
                        "GPU stream desynchronised; requesting a full batch reset"
                    )
                    self.stream_resync_pending = True
                    self.last_status = t("GPU 数据流不同步，正在自动恢复完整视图……")
                    self._update_window_title()
                    try:
                        bridge.submit_resync(str(exc))
                    except GpuEditError:
                        self.running = False
            elif isinstance(patch, GpuBatchResetPatch):
                self._apply_batch_reset(patch)
                self.stream_resync_pending = False
            elif isinstance(patch, GpuStatusPatch):
                self.last_status = patch.message
                self._update_window_title()

    def _ensure_stream_texture(self, upload: GpuTextureUpload) -> int:
        existing = self.texture_key_to_layer.get(upload.key)
        if existing is not None and not upload.replace_existing:
            if existing != upload.layer:
                raise GpuNativeError(
                    f"Texture layer mismatch for {upload.key}: expected {upload.layer}, received {existing}"
                )
            return existing
        if upload.width != self.batch.padded_size or upload.height != self.batch.padded_size:
            raise GpuNativeError(
                f"Stream texture size mismatch: expected {self.batch.padded_size}x{self.batch.padded_size}, "
                f"received {upload.width}x{upload.height}"
            )
        if upload.layer < 0 or upload.layer >= self.maximum_texture_layers:
            raise GpuNativeError(
                f"GPU texture-array layer is outside driver limit: {upload.layer} / {self.maximum_texture_layers}"
            )
        while len(self.texture_pixels) <= upload.layer:
            self.texture_pixels.append(self.texture_placeholder)
            self.texture_sources.append(None)
        source = Path(upload.source_path) if upload.source_path else None
        self.texture_pixels[upload.layer] = upload.pixels_rgba
        self.texture_sources[upload.layer] = source
        reallocated = False
        if len(self.texture_pixels) > self.texture_capacity:
            self.texture_capacity = next_texture_capacity(
                len(self.texture_pixels),
                self.texture_capacity,
                self.maximum_texture_layers,
            )
            self._allocate_texture_storage()
            reallocated = True
        for key, layer in tuple(self.texture_key_to_layer.items()):
            if layer == upload.layer and key != upload.key:
                self.texture_key_to_layer.pop(key, None)
        self.texture_key_to_layer[upload.key] = upload.layer
        if not reallocated:
            self.pending_texture_uploads[upload.layer] = upload.pixels_rgba
        return upload.layer

    def _ensure_instance_capacity(self, required: int) -> bool:
        required = max(1, int(required))
        if required <= self.instance_capacity:
            return False
        self.instance_capacity = _next_power_of_two(required)
        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[1])
        self.gl["glBufferData"](
            self.GL_ARRAY_BUFFER,
            self.instance_capacity * INSTANCE_STRUCT.size,
            None,
            self.GL_DYNAMIC_DRAW,
        )
        return True

    def _upload_instance_slots(self, slots) -> None:
        ordered = sorted({int(slot) for slot in slots if 0 <= int(slot) < len(self.instances)})
        if not ordered:
            return
        self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[1])
        range_start = ordered[0]
        range_end = range_start
        for slot in ordered[1:] + [None]:
            if slot is not None and slot == range_end + 1:
                range_end = slot
                continue
            payload = b"".join(
                self.instances[index].packed()
                for index in range(range_start, range_end + 1)
            )
            buffer = ctypes.create_string_buffer(payload)
            self.gl["glBufferSubData"](
                self.GL_ARRAY_BUFFER,
                range_start * INSTANCE_STRUCT.size,
                len(payload),
                ctypes.cast(buffer, ctypes.c_void_p),
            )
            if slot is None:
                break
            range_start = range_end = slot

    def _upload_all_instances(self) -> None:
        reallocated = self._ensure_instance_capacity(len(self.instances))
        if self.instances:
            self._upload_instance_slots(range(len(self.instances)))
        elif reallocated:
            self.gl["glBindBuffer"](self.GL_ARRAY_BUFFER, self.buffers[1])

    def _apply_stream_patch(self, patch: GpuStreamPatch) -> None:
        for layer in patch.released_texture_layers:
            layer = int(layer)
            self.pending_texture_uploads.pop(layer, None)
            for key, mapped in tuple(self.texture_key_to_layer.items()):
                if mapped == layer and key not in {EMPTY_LAYER_KEY, MISSING_LAYER_KEY}:
                    self.texture_key_to_layer.pop(key, None)
            if 0 <= layer < len(self.texture_pixels):
                self.texture_pixels[layer] = self.texture_placeholder
                self.texture_sources[layer] = None
        for upload in patch.texture_uploads:
            self._ensure_stream_texture(upload)

        dirty_slots: set[int] = set()
        for cell_id in patch.removed_cell_ids:
            slot = self.instance_by_cell.pop(int(cell_id), None)
            if slot is None:
                continue
            last_index = len(self.instances) - 1
            if slot != last_index:
                moved = self.instances[last_index]
                self.instances[slot] = moved
                self.instance_by_cell[moved.cell_id] = slot
                dirty_slots.add(slot)
            self.instances.pop()

        for item in patch.changed:
            if item.slot < 0 or item.slot >= len(self.instances):
                raise GpuStreamDesyncError(
                    f"Changed instance slot is outside current buffer: {item.slot}"
                )
            previous = self.instances[item.slot]
            if previous.cell_id != item.cell_id:
                self.instance_by_cell.pop(previous.cell_id, None)
            self.instances[item.slot] = item.instance
            self.instance_by_cell[item.cell_id] = item.slot
            dirty_slots.add(item.slot)

        for item in patch.added:
            if item.slot != len(self.instances):
                raise GpuStreamDesyncError(
                    f"Added instance slot mismatch: expected {len(self.instances)}, received {item.slot}"
                )
            self.instances.append(item.instance)
            self.instance_by_cell[item.cell_id] = item.slot
            dirty_slots.add(item.slot)

        if len(self.instances) != patch.instance_count:
            raise GpuStreamDesyncError(
                f"Stream instance count mismatch: expected {patch.instance_count}, received {len(self.instances)}"
            )
        reallocated = self._ensure_instance_capacity(len(self.instances))
        if reallocated:
            self._upload_instance_slots(range(len(self.instances)))
        else:
            self._upload_instance_slots(dirty_slots)
        if self.picker is not None:
            self.picker.update_instances(self.instances)
        if self.selected_instance >= len(self.instances):
            self.selected_instance = -1
        self.editable = patch.editable
        self.last_status = patch.message
        if patch.has_more:
            self.force_view_submit = True
            self.view_dirty = True
        self._update_window_title()

    def _apply_batch_reset(self, patch: GpuBatchResetPatch) -> None:
        self.batch = patch.batch
        self._configure_texture_layer_limit(patch.batch.padded_size)
        self.instances = list(patch.batch.instances)
        self.instance_by_cell = {
            instance.cell_id: index for index, instance in enumerate(self.instances)
        }
        side = patch.batch.padded_size
        self.texture_placeholder = bytes((63, 73, 82, 255)) * (side * side)
        self.texture_pixels = []
        self.texture_sources = []
        uploads_by_layer = {
            upload.layer: upload for upload in patch.texture_uploads
        }
        for layer in patch.batch.texture_layers:
            upload = uploads_by_layer.get(layer.layer)
            if upload is not None:
                self.texture_pixels.append(upload.pixels_rgba)
                self.texture_sources.append(
                    Path(upload.source_path)
                    if upload.source_path
                    else layer.source_path
                )
            elif layer.kind == "brush" and layer.source_path is not None:
                self.texture_pixels.append(None)
                self.texture_sources.append(Path(layer.source_path))
            else:
                self.texture_pixels.append(build_texture_layer_payload(layer, side))
                self.texture_sources.append(None)
        self.texture_key_to_layer = {
            layer.key: index for index, layer in enumerate(patch.batch.texture_layers)
        }
        self.texture_capacity = next_texture_capacity(
            len(self.texture_pixels),
            0,
            self.maximum_texture_layers,
        )
        self._allocate_texture_storage()
        self._upload_all_instances()
        if self.picker is not None:
            self.picker.update_instances(self.instances)
        self.selected_instance = -1
        self.editable = patch.editable
        self.last_status = patch.message
        if patch.has_more:
            self.force_view_submit = True
            self.view_dirty = True
        self._update_window_title()

    def _submit_view_if_needed(self) -> None:
        if (
            not self.streaming
            or not self.view_dirty
            or self.edit_bridge is None
            or self.stream_resync_pending
        ):
            return
        now = time.monotonic()
        if not self.force_view_submit and now - self.last_view_submit < 0.08:
            return
        previous = self.last_submitted_view
        size_changed = previous is None or previous[3] != self.width or previous[4] != self.height
        if previous is not None and not self.force_view_submit and not size_changed:
            previous_yaw, previous_pitch, previous_zoom, _, _ = previous
            zoom_ratio = max(self.zoom, previous_zoom) / max(
                1e-6, min(self.zoom, previous_zoom)
            )
            if (
                abs(self.yaw - previous_yaw) < 0.01
                and abs(self.pitch - previous_pitch) < 0.01
                and zoom_ratio < 1.02
            ):
                self.view_dirty = False
                return
        try:
            request_id = self.edit_bridge.submit_view(
                self.yaw,
                self.pitch,
                self.zoom,
                self.width,
                self.height,
                interactive=self.dragging or now < self.interactive_until,
            )
        except GpuEditError as exc:
            self.last_status = str(exc)
            return
        self.last_status = t("已提交视图请求 {request_id}，等待 LOD 流", request_id=request_id)
        self._update_window_title()
        self.last_view_submit = now
        self.last_submitted_view = (
            self.yaw,
            self.pitch,
            self.zoom,
            self.width,
            self.height,
        )
        self.force_view_submit = False
        self.view_dirty = False

    def _update_window_title(self) -> None:
        if not self.hwnd:
            return
        state = self.edit_bridge.tool_state() if self.edit_bridge is not None else None
        if state is None:
            mode = t("只读预览")
        else:
            mode = t("{value} / {value2}°", value=t('放置') if state.tool == 'paint' else t('清除'), value2=state.rotation * 60)
        reserved_bytes = (
            self.texture_capacity * texture_layer_bytes(self.batch.padded_size)
            if self.texture_capacity > 0
            else 0
        )
        texture_status = (
            t("纹理 {len:,}/{maximum_texture_layers:,} 层 · 预留 {value:.2f} GiB · 缩放 {zoom:.2f}×", len=len(self.texture_key_to_layer), maximum_texture_layers=self.maximum_texture_layers, value=reserved_bytes / (1024 ** 3), zoom=self.zoom)
        )
        if self.pending_texture_uploads:
            texture_status += t(" · 待上传 {len:,} 层", len=len(self.pending_texture_uploads))
        text = f"{self.base_title} | {mode} | {texture_status} | {self.last_status}"
        self.user32.SetWindowTextW(self.hwnd, text[:500])

    def _attribute(self, index: int, size: int, stride: int, offset: int, divisor: int) -> None:
        self.gl["glEnableVertexAttribArray"](index)
        self.gl["glVertexAttribPointer"](
            index,
            size,
            self.GL_FLOAT,
            self.GL_FALSE,
            stride,
            ctypes.c_void_p(offset),
        )
        if divisor:
            self.gl["glVertexAttribDivisor"](index, divisor)

    def _compile_shader(self, shader_type: int, source: str) -> int:
        shader = self.gl["glCreateShader"](shader_type)
        encoded = source.encode("utf-8")
        source_pointer = ctypes.c_char_p(encoded)
        length = ctypes.c_int(len(encoded))
        self.gl["glShaderSource"](shader, 1, ctypes.byref(source_pointer), ctypes.byref(length))
        self.gl["glCompileShader"](shader)
        status = ctypes.c_int(0)
        self.gl["glGetShaderiv"](shader, self.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            log = self._shader_log(shader)
            self.gl["glDeleteShader"](shader)
            raise GpuNativeError(f"OpenGL shader compilation failed: {log}")
        return shader

    def _shader_log(self, shader: int) -> str:
        length = ctypes.c_int(0)
        self.gl["glGetShaderiv"](shader, self.GL_INFO_LOG_LENGTH, ctypes.byref(length))
        if length.value <= 1:
            return "no driver log"
        buffer = ctypes.create_string_buffer(length.value)
        self.gl["glGetShaderInfoLog"](shader, length.value, None, buffer)
        return buffer.value.decode("utf-8", errors="replace")

    def _program_log(self, program: int) -> str:
        length = ctypes.c_int(0)
        self.gl["glGetProgramiv"](program, self.GL_INFO_LOG_LENGTH, ctypes.byref(length))
        if length.value <= 1:
            return "no driver log"
        buffer = ctypes.create_string_buffer(length.value)
        self.gl["glGetProgramInfoLog"](program, length.value, None, buffer)
        return buffer.value.decode("utf-8", errors="replace")

    def _uniform(self, name: str) -> int:
        return self._uniform_for(self.program, name)

    def _uniform_for(self, program: int, name: str) -> int:
        location = self.gl["glGetUniformLocation"](program, name.encode("ascii"))
        if location < 0:
            raise GpuNativeError(f"OpenGL uniform was optimized out or missing: {name}")
        return location

    def _message_loop(self) -> None:
        message = MSG()
        while self.running:
            while self.user32.PeekMessageW(ctypes.byref(message), None, 0, 0, self.PM_REMOVE):
                if message.message == 0x0012:
                    self.running = False
                    break
                self.user32.TranslateMessage(ctypes.byref(message))
                self.user32.DispatchMessageW(ctypes.byref(message))
            if not self.running:
                break
            self._poll_edit_bridge()
            self._submit_view_if_needed()
            self._flush_pending_texture_uploads()
            now = time.monotonic()
            if (
                self.frame_dirty
                or self.dragging
                or self.painting
                or now - self.last_render_time >= 0.1
            ):
                self._render()
                self.gdi32.SwapBuffers(self.hdc)
                self.frame_dirty = False
                self.last_render_time = time.monotonic()
                time.sleep(0.001)
            else:
                time.sleep(0.01)

    def _render(self) -> None:
        width = max(1, self.width)
        height = max(1, self.height)
        self.opengl32.glViewport(0, 0, width, height)
        self.opengl32.glClearColor(0.07, 0.09, 0.11, 1.0)
        self.opengl32.glClear(self.GL_COLOR_BUFFER_BIT | self.GL_DEPTH_BUFFER_BIT)
        scale = 0.86 * self.zoom
        self.gl["glUseProgram"](self.sphere_program)
        self.gl["glActiveTexture"](self.GL_TEXTURE1)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D, self.surface_texture.value)
        self.gl["glUniform1f"](self._uniform_for(self.sphere_program, "uYaw"), self.yaw)
        self.gl["glUniform1f"](self._uniform_for(self.sphere_program, "uPitch"), self.pitch)
        self.gl["glUniform1f"](self._uniform_for(self.sphere_program, "uScale"), scale)
        self.gl["glUniform1f"](self._uniform_for(self.sphere_program, "uAspect"), width / height)
        self.gl["glBindVertexArray"](self.sphere_vao.value)
        self.gl["glDrawArraysInstanced"](self.GL_TRIANGLES, 0, self.sphere_vertex_count, 1)

        self.gl["glUseProgram"](self.program)
        self.gl["glUniform1f"](self._uniform("uYaw"), self.yaw)
        self.gl["glUniform1f"](self._uniform("uPitch"), self.pitch)
        self.gl["glUniform1f"](self._uniform("uScale"), scale)
        self.gl["glUniform1f"](self._uniform("uAspect"), width / height)
        self.gl["glUniform1i"](self._uniform("uSelectedInstance"), self.selected_instance)
        self.gl["glBindVertexArray"](self.vao.value)
        self.gl["glActiveTexture"](self.GL_TEXTURE0)
        self.opengl32.glBindTexture(self.GL_TEXTURE_2D_ARRAY, self.texture.value)
        self.gl["glDrawArraysInstanced"](
            self.GL_TRIANGLES,
            0,
            18,
            len(self.instances),
        )

    def _submit_edit_at(
        self, x: int, y: int, *, phase: str, stroke_id: int
    ) -> int | None:
        if self.picker is None or self.edit_bridge is None:
            self.last_status = t("当前窗口没有连接地图编辑会话")
            self._update_window_title()
            return None
        if not self.editable:
            self.selected_instance = -1
            self.last_status = t("当前视图不可编辑")
            self._update_window_title()
            return None
        result = self.picker.pick_screen(
            x, y, self.width, self.height, self.yaw, self.pitch, self.zoom
        )
        if result is None:
            self.selected_instance = -1
            self.last_status = t("未命中星球")
            self._update_window_title()
            return None
        self.selected_instance = -1 if result.instance_index is None else result.instance_index
        if result.cell_id in set(self.picker.topology.pentagon_ids):
            self.last_status = t("CellId {cell_id} 是隐藏五边形，不能编辑", cell_id=result.cell_id)
            self._update_window_title()
            return None
        # A cell outside the current instance stream is still a real map cell:
        # the pick resolves an exact CellId from the topology alone, and the
        # stroke applies to the authoritative Pack session. At the far view the
        # stream is empty by design, so refusing here would make every zoomed-out
        # stroke a no-op. Only the selection highlight needs the instance, and
        # it simply stays off. Feedback for cells that are not streamed arrives
        # when their chunk enters the stream or when the saved surface rebuilds.
        if phase == "move" and result.cell_id == self.last_paint_cell:
            return result.cell_id
        try:
            self.edit_bridge.submit_edit(
                result.cell_id, stroke_id=stroke_id, phase=phase
            )
        except GpuEditError as exc:
            self.last_status = str(exc)
            self._update_window_title()
            return None
        self.last_paint_cell = result.cell_id
        state = self.edit_bridge.tool_state()
        self.last_status = (
            t("连续笔划已提交 CellId {cell_id}：{value}，直径 {brush_diameter} 格", cell_id=result.cell_id, value=t('随机笔刷组覆盖') if state.tool == 'paint' else t('清除'), brush_diameter=state.brush_diameter)
        )
        self._update_window_title()
        return result.cell_id

    def _handle_edit_click(self, x: int, y: int) -> None:
        stroke_id = self.paint_stroke_id + 1
        self._submit_edit_at(x, y, phase="start", stroke_id=stroke_id)
        if self.edit_bridge is not None:
            try:
                self.edit_bridge.submit_stroke_end(stroke_id)
            except GpuEditError:
                pass

    def _handle_key(self, key: int) -> bool:
        bridge = self.edit_bridge
        if key == self.VK_ESCAPE:
            self.user32.PostMessageW(self.hwnd, self.WM_CLOSE, 0, 0)
            return True
        if bridge is None:
            return False
        try:
            if key == self.VK_P:
                bridge.set_tool("paint")
                self.last_status = t("GPU 工具切换为放置")
            elif key == self.VK_E:
                bridge.set_tool("erase")
                self.last_status = t("GPU 工具切换为清除")
            elif key == self.VK_R:
                self.last_status = t("当前笔刷每格自动随机选择 0°～300° 六方向旋转")
            elif key == self.VK_S and self.user32.GetKeyState(self.VK_CONTROL) < 0:
                bridge.submit_save()
                self.last_status = t("已请求保存脏区块")
            else:
                return False
        except GpuEditError as exc:
            self.last_status = str(exc)
        self._update_window_title()
        return True

    def _window_proc(self, hwnd, message, wparam, lparam):
        if message == self.WM_CLOSE:
            self.running = False
            self.user32.DestroyWindow(hwnd)
            return 0
        if message == self.WM_DESTROY:
            self.running = False
            if not self.embedded:
                self.user32.PostQuitMessage(0)
            return 0
        if message == self.WM_SIZE:
            self.width = max(1, lparam & 0xFFFF)
            self.height = max(1, (lparam >> 16) & 0xFFFF)
            self.force_view_submit = self.streaming
            self.view_dirty = self.streaming
            self.frame_dirty = True
            return 0
        if message == self.WM_KEYDOWN and self._handle_key(int(wparam)):
            return 0
        if message == self.WM_LBUTTONDOWN:
            self.user32.SetFocus(hwnd)
            self.painting = True
            self.paint_stroke_id += 1
            self.last_paint_cell = None
            x = _signed_word(lparam & 0xFFFF)
            y = _signed_word((lparam >> 16) & 0xFFFF)
            self.user32.SetCapture(hwnd)
            self._submit_edit_at(
                x, y, phase="start", stroke_id=self.paint_stroke_id
            )
            self.frame_dirty = True
            return 0
        if message == self.WM_LBUTTONUP:
            if self.painting and self.edit_bridge is not None:
                try:
                    self.edit_bridge.submit_stroke_end(self.paint_stroke_id)
                except GpuEditError:
                    pass
            self.painting = False
            self.last_paint_cell = None
            self.user32.ReleaseCapture()
            self.frame_dirty = True
            return 0
        if message == self.WM_RBUTTONDOWN:
            self.user32.SetFocus(hwnd)
            self.dragging = True
            self.last_mouse = (_signed_word(lparam & 0xFFFF), _signed_word((lparam >> 16) & 0xFFFF))
            self.user32.SetCapture(hwnd)
            self.frame_dirty = True
            return 0
        if message == self.WM_RBUTTONUP:
            self.dragging = False
            # Submit one final non-interactive request so the controller can use
            # its larger idle streaming budget after the responsive drag pass.
            self.force_view_submit = self.streaming
            self.view_dirty = self.streaming
            self.user32.ReleaseCapture()
            self.frame_dirty = True
            return 0
        if message == self.WM_MOUSEMOVE and self.dragging:
            x = _signed_word(lparam & 0xFFFF)
            y = _signed_word((lparam >> 16) & 0xFFFF)
            radians_per_pixel = drag_radians_per_pixel(self.zoom)
            self.yaw += (x - self.last_mouse[0]) * radians_per_pixel
            self.pitch = max(
                -1.45,
                min(
                    1.45,
                    self.pitch + (y - self.last_mouse[1]) * radians_per_pixel,
                ),
            )
            self.last_mouse = (x, y)
            self.view_dirty = self.streaming
            self.frame_dirty = True
            return 0
        if message == self.WM_MOUSEMOVE and self.painting:
            x = _signed_word(lparam & 0xFFFF)
            y = _signed_word((lparam >> 16) & 0xFFFF)
            self._submit_edit_at(
                x, y, phase="move", stroke_id=self.paint_stroke_id
            )
            self.frame_dirty = True
            return 0
        if message == self.WM_MOUSEWHEEL:
            delta = _signed_word((wparam >> 16) & 0xFFFF)
            steps = delta / 120.0
            self.zoom = max(self.min_zoom, min(512.0, self.zoom * (1.16**steps)))
            self.view_dirty = self.streaming
            self.interactive_until = time.monotonic() + 0.2
            self.frame_dirty = True
            self._update_window_title()
            return 0
        return self.user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _cleanup(self) -> None:
        if self.edit_bridge is not None:
            self.edit_bridge.close()
        try:
            if self.context and self.hdc:
                self.opengl32.wglMakeCurrent(self.hdc, self.context)
                if self.program and self.gl:
                    self.gl["glDeleteProgram"](self.program)
                if self.sphere_program and self.gl:
                    self.gl["glDeleteProgram"](self.sphere_program)
                if self.texture.value:
                    self.opengl32.glDeleteTextures(1, ctypes.byref(self.texture))
                if self.surface_texture.value:
                    self.opengl32.glDeleteTextures(1, ctypes.byref(self.surface_texture))
                if any(self.buffers) and self.gl:
                    self.gl["glDeleteBuffers"](2, self.buffers)
                if self.sphere_buffer.value and self.gl:
                    self.gl["glDeleteBuffers"](1, ctypes.byref(self.sphere_buffer))
                if self.vao.value and self.gl:
                    self.gl["glDeleteVertexArrays"](1, ctypes.byref(self.vao))
                if self.sphere_vao.value and self.gl:
                    self.gl["glDeleteVertexArrays"](1, ctypes.byref(self.sphere_vao))
                self.opengl32.wglMakeCurrent(None, None)
                self.opengl32.wglDeleteContext(self.context)
        finally:
            if self.hdc and self.hwnd:
                self.user32.ReleaseDC(self.hwnd, self.hdc)
            if self.hwnd and self.user32.IsWindow(self.hwnd):
                self.user32.DestroyWindow(self.hwnd)



def _pixel_image_rgba(image: PixelImage) -> bytes:
    expected = image.width * image.height * image.channels
    if image.width <= 0 or image.height <= 0 or image.channels not in (3, 4):
        raise GpuNativeError("Surface image must be a positive RGB or RGBA image")
    if len(image.pixels) != expected:
        raise GpuNativeError(
            f"Surface image byte count mismatch: expected {expected}, received {len(image.pixels)}"
        )
    if image.channels == 4:
        return image.pixels
    rgba = bytearray(image.width * image.height * 4)
    source = memoryview(image.pixels)
    destination = memoryview(rgba)
    for source_offset in range(0, len(source), 3):
        target_offset = (source_offset // 3) * 4
        destination[target_offset : target_offset + 3] = source[source_offset : source_offset + 3]
        destination[target_offset + 3] = 255
    return bytes(rgba)


def _unit_sphere_mesh_bytes(latitude_segments: int = 36, longitude_segments: int = 72) -> bytes:
    import struct

    vertex = struct.Struct("<3f")
    output: list[bytes] = []
    for latitude in range(latitude_segments):
        phi0 = -math.pi / 2.0 + math.pi * latitude / latitude_segments
        phi1 = -math.pi / 2.0 + math.pi * (latitude + 1) / latitude_segments
        for longitude in range(longitude_segments):
            theta0 = math.tau * longitude / longitude_segments
            theta1 = math.tau * (longitude + 1) / longitude_segments

            def point(phi: float, theta: float) -> tuple[float, float, float]:
                cosine = math.cos(phi)
                return cosine * math.cos(theta), math.sin(phi), cosine * math.sin(theta)

            a = point(phi0, theta0)
            b = point(phi1, theta0)
            c = point(phi1, theta1)
            d = point(phi0, theta1)
            for item in (a, b, c, a, c, d):
                output.append(vertex.pack(*item))
    return b"".join(output)


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < max(1, value):
        result <<= 1
    return result

def _parse_gl_version(value: str) -> tuple[int, int]:
    first = value.split(" ", 1)[0]
    parts = first.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _signed_word(value: int) -> int:
    return ctypes.c_short(value & 0xFFFF).value


def probe_native_gpu_capabilities() -> dict[str, object]:
    """Create a hidden WGL context and compile the real renderer resources.

    This is only executed by the Windows diagnostic tool. On non-Windows
    platforms it returns an explicit unsupported result without pretending that
    WGL was tested.
    """
    if os.name != "nt":
        return {
            "supported": False,
            "executed": False,
            "reason": "WGL diagnostics require Windows",
        }
    empty = GpuRenderBatch(
        version=1,
        lod_level=3,
        effective_size=64,
        padded_size=72,
        topology_hash="0" * 64,
        layout_hash="0" * 64,
        instances=(),
        texture_layers=(
            GpuTextureLayer(
                0, EMPTY_LAYER_KEY, None, "", None, "empty"
            ),
            GpuTextureLayer(
                1, MISSING_LAYER_KEY, None, "", None, "missing"
            ),
        ),
        stable_hash="0" * 64,
    )
    preview = _Win32GpuPreview(
        empty,
        0.0,
        0.0,
        1.0,
        "HexPlanet hidden WGL diagnostic",
        None,
        None,
    )
    preview._register_class(preview.user32, preview.kernel32)
    instance = preview.kernel32.GetModuleHandleW(None)
    preview.hwnd = preview.user32.CreateWindowExW(
        0,
        "HexPlanetGpuPreviewWindow",
        preview.title,
        preview.WS_OVERLAPPEDWINDOW,
        0,
        0,
        64,
        64,
        None,
        None,
        instance,
        None,
    )
    if not preview.hwnd:
        return {
            "supported": False,
            "executed": True,
            "reason": f"CreateWindowExW failed: {preview.kernel32.GetLastError()}",
        }
    preview._registry[int(preview.hwnd)] = preview
    try:
        preview._create_context()
        preview._create_resources()
        version_raw = preview.opengl32.glGetString(preview.GL_VERSION)
        vendor_raw = preview.opengl32.glGetString(0x1F00)
        renderer_raw = preview.opengl32.glGetString(0x1F01)
        max_texture = ctypes.c_int(0)
        max_layers = ctypes.c_int(0)
        preview.opengl32.glGetIntegerv(0x0D33, ctypes.byref(max_texture))
        preview.opengl32.glGetIntegerv(preview.GL_MAX_ARRAY_TEXTURE_LAYERS, ctypes.byref(max_layers))
        return {
            "supported": True,
            "executed": True,
            "openglVersion": "unknown" if not version_raw else version_raw.decode("ascii", "replace"),
            "vendor": "unknown" if not vendor_raw else vendor_raw.decode("ascii", "replace"),
            "renderer": "unknown" if not renderer_raw else renderer_raw.decode("ascii", "replace"),
            "maxTextureSize": int(max_texture.value),
            "maxArrayTextureLayers": int(max_layers.value),
            "shaderCompile": "passed",
            "instancedDrawFunctions": "loaded",
        }
    except Exception as exc:
        return {
            "supported": False,
            "executed": True,
            "reason": str(exc),
        }
    finally:
        preview._cleanup()
        if preview.hwnd:
            preview._registry.pop(int(preview.hwnd), None)
