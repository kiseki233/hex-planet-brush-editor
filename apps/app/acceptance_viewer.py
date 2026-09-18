from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .acceptance import AcceptanceReport, run_final_acceptance
from .paths import ProjectPaths


class AcceptanceViewer:
    def __init__(self, parent: tk.Misc, paths: ProjectPaths) -> None:
        self.paths = paths
        self.window = tk.Toplevel(parent)
        self.window.title("最终验收与 Windows 诊断 v1.3.1")
        self.window.geometry("920x680")
        self.window.minsize(760, 520)
        self.results: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.full_stress = tk.BooleanVar(value=True)
        self.wgl_probe = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="尚未运行")
        self._build_ui()
        self.window.after(100, self._poll)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(root)
        controls.pack(fill=tk.X)
        ttk.Checkbutton(
            controls,
            text="执行 256 张 LOD0 笔刷完整压力测试",
            variable=self.full_stress,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="在 Windows 上执行隐藏 WGL/OpenGL 3.3 实机探测",
            variable=self.wgl_probe,
        ).pack(side=tk.LEFT, padx=(16, 0))
        self.run_button = ttk.Button(controls, text="开始最终验收", command=self.run)
        self.run_button.pack(side=tk.RIGHT)
        self.text = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(10, 8))
        ttk.Label(root, textvariable=self.status, anchor=tk.W, relief=tk.SUNKEN, padding=(8, 4)).pack(fill=tk.X)

    def run(self) -> None:
        if self.running:
            return
        self.running = True
        self.run_button.configure(state=tk.DISABLED)
        self.status.set("正在运行生产索引、Pack、LOD、无缝、纹理压力与 Windows 诊断……")
        self._set_text("验收正在执行。结果会逐项区分通过、失败与需要 Windows 实机确认。\n")

        full_texture_stress = bool(self.full_stress.get())
        run_wgl_probe = bool(self.wgl_probe.get())

        def worker() -> None:
            try:
                report = run_final_acceptance(
                    self.paths,
                    full_texture_stress=full_texture_stress,
                    run_wgl_probe=run_wgl_probe,
                )
                report.write(self.paths.log_root)
                self.results.put(("done", report))
            except Exception as exc:
                self.results.put(("error", exc))

        threading.Thread(target=worker, name="hexplanet-final-acceptance", daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "done":
                    self._show_report(payload)
                else:
                    self.running = False
                    self.run_button.configure(state=tk.NORMAL)
                    self.status.set(f"验收异常：{payload}")
                    messagebox.showerror("最终验收", str(payload), parent=self.window)
        except queue.Empty:
            pass
        if self.window.winfo_exists():
            self.window.after(100, self._poll)

    def _show_report(self, report: AcceptanceReport) -> None:
        self.running = False
        self.run_button.configure(state=tk.NORMAL)
        lines = [
            f"版本：{report.version}",
            f"通过：{report.passed_count}",
            f"失败：{report.failed_count}",
            f"需要 Windows 实机确认：{report.needs_windows_count}",
            "",
        ]
        lines.extend(
            f"[{item.status}] {item.name} ({item.seconds:.3f}s)\n  {item.detail}"
            for item in report.items
        )
        self._set_text("\n\n".join(lines))
        self.status.set(
            f"验收完成：通过 {report.passed_count}，失败 {report.failed_count}，"
            f"需 Windows {report.needs_windows_count}。报告位于 {self.paths.log_root}"
        )

    def _set_text(self, value: str) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", value)
        self.text.configure(state=tk.DISABLED)
