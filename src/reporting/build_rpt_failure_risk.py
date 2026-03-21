from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    return df


def _log_duplicate_sample(df: pd.DataFrame, logger, table_name: str) -> None:
    dupes_mask = df.duplicated(subset=["well_id", "date"], keep=False)
    dupes_count = int(dupes_mask.sum())

    if dupes_count == 0:
        return

    sample = (
        df.loc[dupes_mask]
        .sort_values(["well_id", "date"])
        .head(20)
        .copy()
    )

    logger.warning(
        "%s duplicate well_id/date rows found: %s | sample:\n%s",
        table_name,
        dupes_count,
        sample.to_string(index=False),
    )


def build_rpt_failure_risk(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")
    ml_dir = project_root() / "outputs" / "ml"

    path = ml_dir / "ml_failure_risk_scored.csv"
    if not path.exists():
        logger.warning("ml_failure_risk_scored.csv not found.")
        return pd.DataFrame()

    rpt = pd.read_csv(path, dtype={"well_id": str})
    rpt = _standardize_keys(rpt)

    required = {"well_id", "date"}
    missing = required - set(rpt.columns)
    if missing:
        logger.error("rpt_failure_risk_scored missing required columns: %s", missing)
        return pd.DataFrame()

    bad_dates = int(rpt["date"].isna().sum())
    if bad_dates > 0:
        logger.warning(
            "rpt_failure_risk_scored contains rows with invalid date values: %s. "
            "Dropping rows with null date.",
            bad_dates,
        )
        rpt = rpt.loc[rpt["date"].notna()].copy()

    for col, default in {
        "asset": "UNKNOWN",
        "route": "UNKNOWN",
        "lift_type": "UNKNOWN",
        "equipment_profile_id": "EQP_UNKNOWN",
    }.items():
        if col in rpt.columns:
            rpt[col] = rpt[col].fillna(default)

    _log_duplicate_sample(rpt, logger, "rpt_failure_risk_scored")

    dupes = rpt.duplicated(subset=["well_id", "date"], keep=False)
    if dupes.any():
        rpt = (
            rpt.sort_values(["well_id", "date"])
            .drop_duplicates(subset=["well_id", "date"], keep="last")
            .copy()
        )

        remaining_dupes = rpt.duplicated(subset=["well_id", "date"], keep=False)
        if remaining_dupes.any():
            sample = (
                rpt.loc[remaining_dupes]
                .sort_values(["well_id", "date"])
                .head(20)
            )
            raise ValueError(
                "Unable to resolve duplicate well_id/date rows in rpt_failure_risk_scored.\n"
                + sample.to_string(index=False)
            )

        logger.warning(
            "rpt_failure_risk_scored duplicates were removed by keeping the last row per well_id/date."
        )

    rpt["well_id"] = rpt["well_id"].astype(str).str.strip()
    rpt["date"] = pd.to_datetime(rpt["date"], errors="coerce").dt.date

    write_table(rpt, reports_dir, "rpt_failure_risk_scored", settings)
    batch.set_row_count("rpt_failure_risk_scored", len(rpt))
    logger.info("Built rpt_failure_risk_scored | rows=%s", len(rpt))
    return rpt