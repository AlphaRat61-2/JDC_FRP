from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_chemistry(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    recon_path = modeled_dir / "fact_chem_recon_daily.csv"
    well_path = modeled_dir / "dim_well.csv"
    chem_path = modeled_dir / "dim_chemical.csv"

    if not recon_path.exists():
        logger.warning("fact_chem_recon_daily.csv not found.")
        return pd.DataFrame()

    recon = pd.read_csv(recon_path, dtype={"well_id": str, "chemical_key": str})

    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        recon = recon.merge(
            well[["well_id", "well_name", "asset", "route"]],
            how="left",
            on="well_id",
        )

    if chem_path.exists():
        chem = pd.read_csv(chem_path, dtype={"chemical_key": str})
        recon = recon.merge(
            chem[["chemical_key", "normalized_chemical_name"]],
            how="left",
            on="chemical_key",
        )

    write_table(recon, reports_dir, "rpt_chemistry_daily", settings)
    batch.set_row_count("rpt_chemistry_daily", len(recon))
    logger.info("Built rpt_chemistry_daily | rows=%s", len(recon))
    return recon
