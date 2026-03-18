from __future__ import annotations

import pandas as pd

from src.chemistry.chemical_mapping import map_chemical_names
from src.common.constants import ACTUAL_METHOD_UNKNOWN, CONFIDENCE_UNKNOWN
from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


def build_fact_chem_actual_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    cost_path = staged_dir / "stg_chemical_cost.csv"
    if not cost_path.exists():
        logger.warning("stg_chemical_cost.csv not found.")
        return pd.DataFrame()

    cost = pd.read_csv(cost_path, dtype={"well_id": str})
    if "date" in cost.columns:
        cost["date"] = pd.to_datetime(cost["date"], errors="coerce")

    mapped, exceptions = map_chemical_names(
        cost,
        chem_name_col="chem_name",
        chem_type_col="chem_type",
        table_name="stg_chemical_cost",
        batch_id=batch.batch_id,
    )

    if exceptions:
        append_exceptions(exceptions)

    fact = pd.DataFrame(
        {
            "well_id": mapped["well_id"],
            "chemical_key": mapped["chemical_key"],
            "period_start": mapped["date"].dt.date,
            "period_end": mapped["date"].dt.date,
            "actual_total_volume": mapped["qty"],
            "actual_total_cost": mapped["actual_cost"],
            "actual_unit": "UNKNOWN",
            "source": "chemical_cost",
            "allocation_method": "NONE",
            "actual_confidence": CONFIDENCE_UNKNOWN,
            "actual_method": ACTUAL_METHOD_UNKNOWN,
            "equipment": mapped["equipment"],
            "chem_name_raw": mapped["chem_name"],
            "chem_type_raw": mapped["chem_type"],
        }
    )

    write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
    batch.set_row_count("fact_chem_actual_daily", len(fact))
    logger.info("Built fact_chem_actual_daily | rows=%s", len(fact))
    return fact
