from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_workover(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    wo_path = modeled_dir / "fact_workover_event.csv"
    well_path = modeled_dir / "dim_well.csv"

    if not wo_path.exists():
        logger.warning("fact_workover_event.csv not found.")
        return pd.DataFrame()

    wo = pd.read_csv(wo_path, dtype={"well_id": str})
    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        wo = wo.merge(
            well[["well_id", "well_name", "asset", "route", "lift_type"]],
            how="left",
            on="well_id",
        )

    write_table(wo, reports_dir, "rpt_workover_dashboard", settings)
    batch.set_row_count("rpt_workover_dashboard", len(wo))
    logger.info("Built rpt_workover_dashboard | rows=%s", len(wo))
    return wo
