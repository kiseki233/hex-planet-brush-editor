from __future__ import annotations

import logging
import sys
import tkinter as tk
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import messagebox

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.paths import ProjectPaths
    from app.production_editor import ProductionSphereEditor
else:
    from .paths import ProjectPaths
    from .production_editor import ProductionSphereEditor


def configure_logging(paths: ProjectPaths) -> None:
    paths.log_root.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        paths.log_root / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True,
    )


def main() -> int:
    paths = ProjectPaths.from_app_file(__file__)
    paths.ensure_required_directories()
    configure_logging(paths)

    root = tk.Tk()
    try:
        ProductionSphereEditor(
            root,
            paths,
            main_window=True,
            single_map_name="planet",
            auto_open_gpu=False,
        )
        root.mainloop()
        return 0
    except Exception as exc:
        logging.exception("Unhandled application error")
        try:
            messagebox.showerror("错误", f"程序发生未处理错误：\n{exc}\n\n日志：{paths.log_root / 'app.log'}")
        finally:
            root.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
