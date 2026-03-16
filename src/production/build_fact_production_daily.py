from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_fact_production_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    path = staged_dir / "stg_production_daily.csv"
    if not path.exists():
        logger.warning("stg_production_daily.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"], dtype={"well_id": str})

    fact = df[["well_id", "date", "oil_bbl", "gas_mcf", "water_bbl", "source_system"]].copy()
    fact["total_fluid_bbl"] = fact["oil_bbl"].fillna(0) + fact["water_bbl"].fillna(0)
    gas_factor = settings["units"]["boe_gas_mcf_per_boe"]
    fact["boe"] = fact["oil_bbl"].fillna(0) + (fact["gas_mcf"].fillna(0) / gas_factor)
    fact["prod_data_status"] = "MEASURED"
    fact["prod_source"] = fact["source_system"]
    fact = fact.drop(columns=["source_system"])
    fact["date"] = fact["date"].dt.date

    write_table(fact, modeled_dir, "fact_production_daily", settings)
    batch.set_row_count("fact_production_daily", len(fact))
    logger.info("Built fact_production_daily | rows=%s", len(fact))
    return fact