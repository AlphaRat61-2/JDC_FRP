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


def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    return df


def _log_duplicate_keys(df: pd.DataFrame, logger, name: str) -> None:
    if not {"well_id", "date"}.issubset(df.columns):
        return

    dupes = df.duplicated(subset=["well_id", "date"], keep=False)
    count = int(dupes.sum())

    if count > 0:
        sample = (
            df.loc[dupes]
            .sort_values(["well_id", "date"])
            .head(20)
            .copy()
        )
        logger.warning(
            "%s duplicate well_id/date rows found: %s | sample:\n%s",
            name,
            count,
            sample.to_string(index=False),
        )


def _assert_required_columns(df: pd.DataFrame, cols: list[str], logger, name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error("%s missing required columns: %s", name, missing)
        return False
    return True


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
    feat = _standardize_keys(feat)

    if not _assert_required_columns(feat, ["well_id", "date"], logger, "ml_feature_well_daily"):
        return pd.DataFrame()

    bad_feat_dates = int(feat["date"].isna().sum())
    if bad_feat_dates > 0:
        logger.warning(
            "ml_feature_well_daily contains rows with invalid date values: %s. "
            "Dropping rows with null date before scoring.",
            bad_feat_dates,
        )
        feat = feat.loc[feat["date"].notna()].copy()

    _log_duplicate_keys(feat, logger, "ml_feature_well_daily")

    feat_dupes = feat.duplicated(subset=["well_id", "date"], keep=False)
    if feat_dupes.any():
        feat = (
            feat.sort_values(["well_id", "date"])
            .drop_duplicates(subset=["well_id", "date"], keep="last")
            .copy()
        )
        logger.warning(
            "ml_feature_well_daily duplicates were removed by keeping the last row per well_id/date before scoring."
        )

    model = joblib.load(model_path)

    numeric_cols = [c for c in FEATURE_COLUMNS if c in feat.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in feat.columns]
    feature_cols = numeric_cols + categorical_cols

    if not feature_cols:
        logger.error("No model feature columns found in ml_feature_well_daily.")
        return pd.DataFrame()

    score_df = feat.copy()
    score_df["failure_risk_30d"] = model.predict_proba(score_df[feature_cols])[:, 1]

    score_df["risk_bucket"] = pd.cut(
        score_df["failure_risk_30d"],
        bins=[-0.01, 0.2, 0.5, 0.75, 1.0],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"],
    )

    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        well = _standardize_keys(well)

        if "well_id" not in well.columns:
            logger.warning("dim_well.csv found but missing well_id column. Skipping well enrichment.")
        else:
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
            ].copy()

            well_dupes = well.duplicated(subset=["well_id"], keep=False)
            if well_dupes.any():
                sample = (
                    well.loc[well_dupes]
                    .sort_values(["well_id"])
                    .head(20)
                    .copy()
                )
                logger.warning(
                    "dim_well duplicate well_id rows found: %s | sample:\n%s",
                    int(well_dupes.sum()),
                    sample.to_string(index=False),
                )
                well = (
                    well.sort_values(["well_id"])
                    .drop_duplicates(subset=["well_id"], keep="last")
                    .copy()
                )
                logger.warning(
                    "dim_well duplicates were removed by keeping the last row per well_id before merge."
                )

            score_df = score_df.merge(
                well,
                how="left",
                on="well_id",
                suffixes=("", "_well"),
                validate="many_to_one",
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

    score_df = _standardize_keys(score_df)
    _log_duplicate_keys(score_df, logger, "ml_failure_risk_scored_pre_write")

    remaining_dupes = score_df.duplicated(subset=["well_id", "date"], keep=False)
    if remaining_dupes.any():
        sample = (
            score_df.loc[remaining_dupes]
            .sort_values(["well_id", "date"])
            .head(20)
            .copy()
        )
        raise ValueError(
            "Duplicate well_id/date rows still exist in ml_failure_risk_scored before write.\n"
            + sample.to_string(index=False)
        )

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