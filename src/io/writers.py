from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_table(
    df: pd.DataFrame,
    base_path: Path,
    table_name: str,
    settings: Dict[str, Any],
) -> None:
    if settings["runtime"].get("write_csv", True):
        write_csv(df, base_path / f"{table_name}.csv")
    if settings["runtime"].get("write_parquet", False):
        write_parquet(df, base_path / f"{table_name}.parquet")