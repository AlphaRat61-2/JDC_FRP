from __future__ import annotations

import joblib
import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table
from src.ml.model_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def _ml_dir():
    path = project_root() / "outputs" / "ml"
    path.mkdir(parents=True, exist_ok=True)
    return path


def score_failure_risk(settings, logger, batch) -> pd.DataFrame:
    ml_dir = _ml_dir()
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    feat_path = ml_dir / "ml_feature_well_daily.csv"
    model_path = ml_dir / "baseline_failure_model.joblib"
    well_path = modeled_dir / "dim_well.csv"

    if not feat_path.exists() or not model_path.exists():
        logger.warning("Missing feature table or trained model for scoring.")
        return pd.DataFrame()

    feat = pd.read_csv(feat_path, parse_dates=["date"], dtype={"well_id": str})
    model = joblib.load(model_path)

    numeric_cols = [c for c in FEATURE_COLUMNS if c in feat.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in feat.columns]
    feature_cols = numeric_cols + categorical_cols

    score_df = feat.copy()
    score_df["failure_risk_30d"] = model.predict_proba(score_df[feature_cols])[:, 1]

    score_df["risk_bucket"] = pd.cut(
        score_df["failure_risk_30d"],
        bins=[-0.01, 0.2, 0.5, 0.75, 1.0],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"],
    )

    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})

        for col, default in {
            "well_name": "UNKNOWN",
            "asset": "UNKNOWN",
            "route": "UNKNOWN",
            "lift_type": "UNKNOWN",
            "equipment_profile_id": "EQP_UNKNOWN",
        }.items():
            if col not in well.columns:
                well[col] = default

        well = well[
            ["well_id", "well_name", "asset", "route", "lift_type", "equipment_profile_id"]
        ].drop_duplicates()

        score_df = score_df.merge(
            well,
            how="left",
            on="well_id",
            suffixes=("", "_well"),
        )

        for col, default in {
            "well_name": "UNKNOWN",
            "asset": "UNKNOWN",
            "route": "UNKNOWN",
            "lift_type": "UNKNOWN",
            "equipment_profile_id": "EQP_UNKNOWN",
        }.items():
            well_col = f"{col}_well"
            if col not in score_df.columns:
                score_df[col] = default
            if well_col in score_df.columns:
                score_df[col] = score_df[col].fillna(score_df[well_col])
                score_df = score_df.drop(columns=[well_col])

    for col, default in {
        "well_name": "UNKNOWN",
        "asset": "UNKNOWN",
        "route": "UNKNOWN",
        "lift_type": "UNKNOWN",
        "equipment_profile_id": "EQP_UNKNOWN",
    }.items():
        if col not in score_df.columns:
            score_df[col] = default
        score_df[col] = score_df[col].fillna(default)

    score_df["date"] = pd.to_datetime(score_df["date"], errors="coerce").dt.date
    score_df["high_risk_flag"] = score_df["risk_bucket"].isin(["HIGH", "CRITICAL"]).astype(int)
    score_df["risk_rank_daily"] = (
        score_df.groupby("date")["failure_risk_30d"]
        .rank(method="dense", ascending=False)
    )

    keep_cols = [
        "well_id",
        "well_name",
        "asset",
        "route",
        "lift_type",
        "equipment_profile_id",
        "date",
        "failure_risk_30d",
        "risk_bucket",
        "high_risk_flag",
        "risk_rank_daily",
        "days_since_last_failure",
        "failures_last_90d",
        "chem_exception_7d",
        "runtime_7d_avg",
        "fillage_7d_avg",
        "trip_count_7d",
        "shutdown_count_7d",
        "deterioration_score",
        "pre_failure_flag",
    ]
    keep_cols = [c for c in keep_cols if c in score_df.columns]
    score_df = score_df[keep_cols].copy()

    write_table(score_df, ml_dir, "ml_failure_risk_scored", settings)
    write_table(score_df, reports_dir, "rpt_failure_risk_scored", settings)

    batch.set_row_count("ml_failure_risk_scored", len(score_df))
    batch.set_row_count("rpt_failure_risk_scored", len(score_df))
    logger.info("Built failure risk scoring outputs | rows=%s", len(score_df))
    return score_df