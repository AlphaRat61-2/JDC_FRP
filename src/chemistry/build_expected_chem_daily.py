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

    program = load_well_chem_program()
    daily, exceptions = expand_well_chem_program_daily(program, batch_id=batch.batch_id)

    if exceptions:
        append_exceptions(settings, exceptions, logger)

    write_table(daily, modeled_dir, "fact_expected_chem_daily", settings)
    batch.set_row_count("fact_expected_chem_daily", len(daily))
    logger.info("Built fact_expected_chem_daily | rows=%s", len(daily))
    return daily
