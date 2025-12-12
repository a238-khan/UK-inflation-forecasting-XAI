from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

DEFAULT_DATA_PATHS = {
    "model_ready": Path("notebooks/model_ready_dataset.csv"),
}


def find_project_root(marker: str = "data") -> Path:
    cwd = Path.cwd()
    for path in (cwd, *cwd.parents):
        if (path / marker).exists():
            return path
    return cwd


def load_model_ready_dataset(paths: Dict[str, Path] = None) -> pd.DataFrame:
    paths = paths or DEFAULT_DATA_PATHS
    root = find_project_root()
    csv_path = root / paths["model_ready"]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values("date").reset_index(drop=True)
    return df
