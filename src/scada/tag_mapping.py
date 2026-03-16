from __future__ import annotations

import pandas as pd

from src.common.paths import project_root


def load_scada_tag_map() -> pd.DataFrame:
    path = project_root() / "config" / "scada_tag_map.csv"
    if path.exists():
        df = pd.read_csv(path)
        required = {"tag_name", "lift_type", "metric_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"scada_tag_map.csv missing columns: {sorted(missing)}")
        return df
    return pd.DataFrame(columns=["tag_name", "lift_type", "metric_name"])


def map_scada_tags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    tag_map = load_scada_tag_map()
    if tag_map.empty:
        out = df.copy()
        out["metric_name"] = None
        return out

    out = df.merge(tag_map, how="left", on=["tag_name", "lift_type"])
    return out
