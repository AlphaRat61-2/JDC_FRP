from __future__ import annotations

import pandas as pd

from src.common.paths import project_root


def load_chem_tolerance() -> pd.DataFrame:
    path = project_root() / "config" / "chem_tolerance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        {
            "chem_type": ["default"],
            "low_variance_pct": [10.0],
            "medium_variance_pct": [20.0],
            "high_variance_pct": [35.0],
        }
    )


def get_tolerance_row(chem_type: str | None, tol_df: pd.DataFrame) -> pd.Series:
    if chem_type and chem_type in tol_df["chem_type"].values:
        return tol_df[tol_df["chem_type"] == chem_type].iloc[0]
    return tol_df[tol_df["chem_type"] == "default"].iloc[0]


def classify_operational_status(
    *,
    target_rate: float | None,
    actual_rate: float | None,
    chem_type: str | None,
    tol_df: pd.DataFrame,
) -> str:
    if pd.isna(target_rate) and pd.isna(actual_rate):
        return "UNKNOWN"
    if not pd.isna(target_rate) and pd.isna(actual_rate):
        return "TARGET_ONLY"
    if pd.isna(target_rate) and not pd.isna(actual_rate):
        return "ACTUAL_ONLY"
    if pd.isna(target_rate) or target_rate == 0:
        return "UNKNOWN"

    variance_pct = ((actual_rate - target_rate) / target_rate) * 100.0
    tol = get_tolerance_row(chem_type, tol_df)
    low = float(tol["low_variance_pct"])

    if variance_pct < -low:
        return "UNDER"
    if variance_pct > low:
        return "OVER"
    return "COMPLIANT"
