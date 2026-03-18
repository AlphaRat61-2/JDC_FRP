from __future__ import annotations

import pandas as pd

from src.common.paths import project_root, get_path
from src.io.writers import write_table


def build_rpt_top_risk_wells(settings, logger, batch) -> pd.DataFrame:
    ml_dir = project_root() / "outputs" / "ml"
    reports_dir = get_path(settings, "reports")

    path = ml_dir / "ml_failure_risk_scored.csv"
    if not path.exists():
        logger.warning("ml_failure_risk_scored.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"], dtype={"well_id": str})

    # -----------------------------------------------------
    # Get latest date only
    # -----------------------------------------------------
    latest_date = df["date"].max()
    df = df[df["date"] == latest_date].copy()

    # -----------------------------------------------------
    # Sort by highest risk
    # -----------------------------------------------------
    df = df.sort_values("failure_risk_30d", ascending=False)

    # -----------------------------------------------------
    # Top 25 wells
    # -----------------------------------------------------
    top = df.head(25).copy()

    # -----------------------------------------------------
    # Add explanation flags
    # -----------------------------------------------------
    top["flag_low_runtime"] = (top["runtime_7d_avg"] < 12).astype(int)
    top["flag_high_trips"] = (top["trip_count_7d"] > 5).astype(int)
    top["flag_high_shutdowns"] = (top["shutdown_count_7d"] > 3).astype(int)
    top["flag_high_chem_exceptions"] = (top["chem_exception_7d"] > 0).astype(int)
    top["flag_recent_failures"] = (top["failures_last_90d"] > 0).astype(int)
    top["flag_deterioration"] = (top["deterioration_score"] > 0).astype(int)

    # simple score = count of issues
    top["risk_driver_count"] = (
        top[
            [
                "flag_low_runtime",
                "flag_high_trips",
                "flag_high_shutdowns",
                "flag_high_chem_exceptions",
                "flag_recent_failures",
                "flag_deterioration",
            ]
        ].sum(axis=1)
    )

    write_table(top, reports_dir, "rpt_top_risk_wells", settings)
    batch.set_row_count("rpt_top_risk_wells", len(top))

    logger.info("Built rpt_top_risk_wells | rows=%s", len(top))
    return top