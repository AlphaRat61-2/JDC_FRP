from __future__ import annotations

import pandas as pd


def add_surveillance_features(df: pd.DataFrame, *, lift_type: str) -> pd.DataFrame:
    out = df.copy()
    out["deterioration_score"] = 0.0
    out["pre_failure_flag"] = False

    if lift_type == "ROD_PUMP":
        out.loc[out["avg_fillage_pct"].fillna(100) < 55, "deterioration_score"] += 1.0
        out.loc[out["shutdown_count"].fillna(0) > 3, "deterioration_score"] += 1.0
        out.loc[out["runtime_hr"].fillna(24) < 12, "deterioration_score"] += 1.0

    if lift_type == "ESP":
        out.loc[out["trip_count"].fillna(0) > 2, "deterioration_score"] += 1.0
        out.loc[out["shutdown_count"].fillna(0) > 3, "deterioration_score"] += 1.0
        out.loc[out["runtime_hr"].fillna(24) < 12, "deterioration_score"] += 1.0
        out.loc[out["max_motor_temp"].fillna(0) > 220, "deterioration_score"] += 1.0

    out["pre_failure_flag"] = out["deterioration_score"] >= 2.0
    return out
