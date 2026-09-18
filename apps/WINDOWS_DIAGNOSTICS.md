# Windows and WGL Diagnostics

Run from the main window or command line:

```powershell
pwsh.exe -File apps/tools/run_windows_diagnostic.ps1
```

The diagnostic checks:

- Python 3.10 or newer.
- tkinter/Tcl availability.
- PowerShell 7 and Windows PowerShell fallback.
- `start.bat` and `start.ps1` presence.
- Windows video-controller names.
- Hidden Win32/WGL context creation.
- OpenGL version, vendor, and renderer.
- Shader/program setup through the real native renderer path.
- `GL_MAX_TEXTURE_SIZE`.
- `GL_MAX_ARRAY_TEXTURE_LAYERS`.

Reports:

```text
apps/runtime/logs/windows_diagnostic.json
apps/runtime/logs/windows_diagnostic.txt
```

On a non-Windows system, WGL items are marked `needs_windows`, not passed and not failed.
