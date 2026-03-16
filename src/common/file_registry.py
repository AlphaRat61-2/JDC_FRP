from pathlib import Path
from datetime import datetime
import pandas as pd

from src.common.paths import ROOT


REGISTRY_FILE = ROOT / "data" / "reports" / "file_registry.csv"


def _empty_registry():

    return pd.DataFrame(columns=[
        "file_name",
        "file_path",
        "file_type",
        "stage",
        "created_timestamp",
        "row_count",
        "status"
    ])


def load_registry():

    if REGISTRY_FILE.exists():
        return pd.read_csv(REGISTRY_FILE)

    return _empty_registry()


def register_file(file_path, file_type, stage, row_count=None, status="created"):

    file_path = Path(file_path)

    registry = load_registry()

    row = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_type": file_type,
        "stage": stage,
        "created_timestamp": datetime.utcnow().isoformat(),
        "row_count": row_count,
        "status": status
    }

    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)

    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)

    registry.to_csv(REGISTRY_FILE, index=False)

    return registry


def update_status(file_path, status):

    registry = load_registry()

    mask = registry["file_path"] == str(file_path)

    registry.loc[mask, "status"] = status

    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)

    registry.to_csv(REGISTRY_FILE, index=False)


def get_files_by_stage(stage):

    registry = load_registry()

    return registry[registry["stage"] == stage]


def get_latest_file(file_type):

    registry = load_registry()

    df = registry[registry["file_type"] == file_type]

    if df.empty:
        return None

    df = df.sort_values("created_timestamp", ascending=False)

    return df.iloc[0]["file_path"]


def summarize_registry():

    registry = load_registry()

    if registry.empty:
        return registry

    summary = (
        registry.groupby(["stage", "file_type"])
        .size()
        .reset_index(name="file_count")
        .sort_values("file_count", ascending=False)
    )

    return summary


def clear_registry():

    if REGISTRY_FILE.exists():
        REGISTRY_FILE.unlink()