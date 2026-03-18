from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def build_rpt_failure_risk(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")
    ml_dir = project_root() / "outputs" / "ml"

    path = ml_dir / "ml_failure_risk_scored.csv"
    if not path.exists():
        logger.warning("ml_failure_risk_scored.csv not found.")
        return pd.DataFrame()

    rpt = pd.read_csv(path, dtype={"well_id": str})

    required = {"well_id", "date"}
    missing = required - set(rpt.columns)
    if missing:
        logger.error("rpt_failure_risk_scored missing required columns: %s", missing)
        return pd.DataFrame()

    rpt["well_id"] = rpt["well_id"].astype(str)
    rpt["date"] = pd.to_datetime(rpt["date"], errors="coerce").dt.date

    for col, default in {
        "asset": "UNKNOWN",
        "route": "UNKNOWN",
        "lift_type": "UNKNOWN",
        "equipment_profile_id": "EQP_UNKNOWN",
    }.items():
        if col in rpt.columns:
            rpt[col] = rpt[col].fillna(default)

    dupes = rpt.duplicated(subset=["well_id", "date"]).sum()
    if dupes > 0:
        logger.warning("rpt_failure_risk_scored duplicate well_id/date rows found: %s", dupes)
        rpt = rpt.sort_values(["well_id", "date"]).drop_duplicates(
            subset=["well_id", "date"],
            keep="last",
        )

    write_table(rpt, reports_dir, "rpt_failure_risk_scored", settings)
    batch.set_row_count("rpt_failure_risk_scored", len(rpt))
    logger.info("Built rpt_failure_risk_scored | rows=%s", len(rpt))
    return rpt
