from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the repo root by walking up until a folder containing 'src' exists."""
    cur = (start or Path.cwd()).resolve()
    while cur != cur.parent:
        if (cur / "src").exists():
            return cur
        cur = cur.parent
    return (start or Path.cwd()).resolve()


def get_run_id() -> str:
    """Return a stable run id for the current process.

    Priority:
    1) Environment variable RUN_ID
    2) Current timestamp
    """

    rid = os.environ.get("RUN_ID")
    if rid:
        return rid
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    run_id: str
    notebook_name: str

    @property
    def root(self) -> Path:
        return self.project_root / "artifacts" / "runs" / self.run_id / self.notebook_name

    @property
    def figures(self) -> Path:
        return self.root / "figures"


def get_run_paths(notebook_name: str, project_root: Optional[Path] = None) -> RunPaths:
    root = project_root or find_project_root()
    return RunPaths(project_root=root, run_id=get_run_id(), notebook_name=notebook_name)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_all_matplotlib_figures(out_dir: Path, prefix: str = "fig", dpi: int = 220) -> list[Path]:
    """Save all currently-open matplotlib figures into out_dir.

    Returns the list of written file paths.
    """

    import matplotlib.pyplot as plt

    ensure_dirs(out_dir)

    written: list[Path] = []
    fignums = list(plt.get_fignums())
    for i, num in enumerate(fignums, start=1):
        fig = plt.figure(num)
        out_path = out_dir / f"{prefix}_{i:02d}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        written.append(out_path)

    return written
