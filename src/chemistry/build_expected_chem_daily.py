from __future__ import annotations

import pandas as pd

from src.chemistry.well_chem_program import (
    expand_well_chem_program_daily,
    load_well_chem_program,
)
from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


def build_expected_chem_daily(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")

    # ------------------------------------------------------------
    # Load chemical program
    # ------------------------------------------------------------
    program = load_well_chem_program()

    # ------------------------------------------------------------
    # Guard: no program → no expected chemistry
    # ------------------------------------------------------------
    if program.empty:
        logger.warning("Expected chemistry cannot be built — well_chem_program is empty.")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Guard: missing required columns
    # ------------------------------------------------------------
    required_cols = {"well_id", "chemical_key", "start_date", "end_date"}
    missing = required_cols - set(program.columns)

    if missing:
        logger.error(f"Expected chemistry cannot be built — missing columns: {missing}")
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Expand to daily rows
    # ------------------------------------------------------------
    daily, exceptions = expand_well_chem_program_daily(program, batch_id=batch.batch_id)

    if exceptions:
        append_exceptions(exceptions)

    write_table(daily, modeled_dir, "fact_expected_chem_daily", settings)
    batch.set_row_count("fact_expected_chem_daily", len(daily))
    logger.info("Built fact_expected_chem_daily | rows=%s", len(daily))
    return daily
