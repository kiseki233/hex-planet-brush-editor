from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .gpu_native import native_gpu_supported, probe_native_gpu_capabilities
from .paths import ProjectPaths


@dataclass(frozen=True)
class DiagnosticItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class WindowsDiagnosticReport:
    platform: str
    items: tuple[DiagnosticItem, ...]
    gpu: dict[str, object]

    @property
    def failed_count(self) -> int:
        return sum(item.status == "fail" for item in self.items)

    @property
    def needs_windows_count(self) -> int:
        return sum(item.status == "needs_windows" for item in self.items)

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "windows_diagnostic.json"
        text_path = directory / "windows_diagnostic.txt"
        json_path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"Platform: {self.platform}", ""]
        lines.extend(f"[{item.status}] {item.name}: {item.detail}" for item in self.items)
        lines.append("")
        lines.append("GPU/WGL:")
        lines.extend(f"  {key}: {value}" for key, value in self.gpu.items())
        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, text_path


def collect_windows_diagnostics(paths: ProjectPaths, *, run_wgl_probe: bool = True) -> WindowsDiagnosticReport:
    items: list[DiagnosticItem] = []
    items.append(
        DiagnosticItem(
            "Python",
            "pass" if sys.version_info >= (3, 10) else "fail",
            f"{sys.version.split()[0]} at {sys.executable}",
        )
    )
    try:
        import tkinter

        tk_detail = f"Tk {tkinter.TkVersion} / Tcl {tkinter.TclVersion}"
        tk_status = "pass"
    except Exception as exc:
        tk_detail = str(exc)
        tk_status = "fail"
    items.append(DiagnosticItem("tkinter", tk_status, tk_detail))

    pwsh = shutil.which("pwsh.exe") or shutil.which("pwsh")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    items.append(
        DiagnosticItem(
            "PowerShell 7",
            "pass" if pwsh else "info",
            pwsh or "not found; launcher will fall back when Windows PowerShell exists",
        )
    )
    items.append(
        DiagnosticItem(
            "Windows PowerShell fallback",
            "pass" if powershell else ("needs_windows" if os.name != "nt" else "fail"),
            powershell or "not available in the current environment",
        )
    )
    launchers = [paths.apps_root / "start.bat", paths.apps_root / "start.ps1"]
    items.append(
        DiagnosticItem(
            "Launch scripts",
            "pass" if all(path.is_file() for path in launchers) else "fail",
            ", ".join(str(path) for path in launchers),
        )
    )

    if os.name == "nt":
        gpu_names = _windows_gpu_names()
        items.append(
            DiagnosticItem(
                "Windows GPU inventory",
                "pass" if gpu_names else "info",
                "; ".join(gpu_names) if gpu_names else "GPU name could not be read",
            )
        )
        if run_wgl_probe:
            gpu = probe_native_gpu_capabilities()
            items.append(
                DiagnosticItem(
                    "Hidden WGL/OpenGL 3.3 probe",
                    "pass" if gpu.get("supported") else "fail",
                    str(gpu.get("openglVersion") or gpu.get("reason") or "unknown"),
                )
            )
        else:
            gpu = {"supported": native_gpu_supported(), "executed": False, "reason": "probe disabled"}
            items.append(DiagnosticItem("Hidden WGL/OpenGL 3.3 probe", "info", "probe disabled"))
    else:
        gpu = probe_native_gpu_capabilities()
        items.append(
            DiagnosticItem(
                "Windows WGL/OpenGL 3.3 probe",
                "needs_windows",
                "current platform is not Windows; run the same diagnostic on the target PC",
            )
        )

    return WindowsDiagnosticReport(
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        items=tuple(items),
        gpu=gpu,
    )


def _windows_gpu_names() -> tuple[str, ...]:
    commands = [
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ],
        ["wmic.exe", "path", "win32_VideoController", "get", "name"],
    ]
    for command in commands:
        executable = shutil.which(command[0])
        if not executable:
            continue
        command[0] = executable
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except Exception:
            continue
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        lines = [line for line in lines if line.lower() != "name"]
        if lines:
            return tuple(lines)
    return ()
