from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_scada(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    scada_path = modeled_dir / "fact_scada_daily.csv"
    well_path = modeled_dir / "dim_well.csv"

    if not scada_path.exists():
        logger.warning("fact_scada_daily.csv not found.")
        return pd.DataFrame()

    scada = pd.read_csv(scada_path, dtype={"well_id": str})
    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        scada = scada.merge(
            well[["well_id", "well_name", "asset", "route", "lift_type"]],
            how="left",
            on="well_id",
            suffixes=("", "_well"),
        )

    write_table(scada, reports_dir, "rpt_scada_surveillance_daily", settings)
    batch.set_row_count("rpt_scada_surveillance_daily", len(scada))
    logger.info("Built rpt_scada_surveillance_daily | rows=%s", len(scada))
    return scada
