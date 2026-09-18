from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    apps_root: Path
    art_root: Path
    brush_root: Path
    map_root: Path
    runtime_root: Path
    log_root: Path

    @classmethod
    def from_app_file(cls, app_file: str | Path) -> "ProjectPaths":
        app_dir = Path(app_file).resolve().parent
        apps_root = app_dir.parent
        project_root = apps_root.parent
        art_root = project_root / "art"
        return cls(
            project_root=project_root,
            apps_root=apps_root,
            art_root=art_root,
            brush_root=art_root / "brushes",
            map_root=art_root / "maps",
            runtime_root=apps_root / "runtime",
            log_root=apps_root / "runtime" / "logs",
        )

    @property
    def data_root(self) -> Path:
        return self.art_root / "data"

    def ensure_required_directories(self) -> None:
        self.brush_root.mkdir(parents=True, exist_ok=True)
        self.map_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
