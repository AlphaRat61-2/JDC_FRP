from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_fact_failure_event(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    path = staged_dir / "stg_failures.csv"
    if not path.exists():
        logger.warning("stg_failures.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["install_date", "fail_date"], dtype={"well_id": str})

    fact = df.copy()
    fact["failure_event_id"] = (
        fact["well_id"].astype(str)
        + "_"
        + fact["install_date"].dt.strftime("%Y%m%d").fillna("NA")
        + "_"
        + fact["fail_date"].dt.strftime("%Y%m%d").fillna("NA")
    )
    fact["run_life_days"] = (fact["fail_date"] - fact["install_date"]).dt.days
    fact["failure_source"] = "failures_file"

    keep_cols = [
        "failure_event_id", "well_id", "install_date", "fail_date",
        "equipment_type", "failure_type", "failure_cause",
        "failure_location", "depth", "failure_cost", "vendor",
        "comment", "run_life_days", "failure_source"
    ]
    fact = fact[keep_cols].copy()
    fact["install_date"] = fact["install_date"].dt.date
    fact["fail_date"] = fact["fail_date"].dt.date

    write_table(fact, modeled_dir, "fact_failure_event", settings)
    batch.set_row_count("fact_failure_event", len(fact))
    logger.info("Built fact_failure_event | rows=%s", len(fact))
    return fact