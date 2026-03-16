from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9_]+", "", col.strip().lower().replace(" ", "_"))
        for col in df.columns
    ]
    return df


def read_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return standardize_column_names(df)


def add_source_metadata(
    df: pd.DataFrame,
    file_name: str,
    batch_id: str,
    source_name: str,
) -> pd.DataFrame:
    df = df.copy()
    df["source_file_name"] = file_name
    df["batch_id"] = batch_id
    df["source_system"] = source_name
    return df