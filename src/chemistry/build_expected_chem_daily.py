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

    target_path = modeled_dir / "fact_chem_target_daily.csv"

    # ------------------------------------------------------------
    # Preferred source: daily target table
    # ------------------------------------------------------------
    if target_path.exists():
        target = pd.read_csv(target_path, dtype={"well_id": str, "chemical_key": str})
        if not target.empty:
            if "date" in target.columns:
                target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.date

            expected = pd.DataFrame(
                {
                    "well_id": target["well_id"],
                    "date": target["date"],
                    "chemical_key": target["chemical_key"],
                    "expected_rate": pd.to_numeric(target["target_rate"], errors="coerce"),
                    "expected_unit": target.get("target_unit", "GPD"),
                    "expected_basis": target.get("target_basis"),
                    "expected_source": target.get("target_source", "chemical_rates"),
                    "expected_source_priority": target.get("target_source_priority", 3),
                    "expected_confidence": target.get("target_confidence", "HIGH"),
                    "expected_status": target.get("target_status", "ACTIVE"),
                }
            ).drop_duplicates()

            write_table(expected, modeled_dir, "fact_expected_chem_daily", settings)
            batch.set_row_count("fact_expected_chem_daily", len(expected))
            logger.info(
                "Built fact_expected_chem_daily from fact_chem_target_daily | rows=%s",
                len(expected),
            )
            return expected

    # ------------------------------------------------------------
    # Fallback source: interval program config
    # ------------------------------------------------------------
    program = load_well_chem_program()

    if program.empty:
        logger.warning("Expected chemistry cannot be built — no daily target or chem program found.")
        return pd.DataFrame()

    daily, exceptions = expand_well_chem_program_daily(program, batch_id=batch.batch_id)

    if exceptions:
        append_exceptions(exceptions)

    if daily.empty:
        logger.warning("Expected chemistry cannot be built — chem program expansion returned no rows.")
        return pd.DataFrame()

    expected = pd.DataFrame(
        {
            "well_id": daily["well_id"],
            "date": pd.to_datetime(daily["date"], errors="coerce").dt.date,
            "chemical_key": daily["chemical_key"],
            "expected_rate": pd.NA,
            "expected_unit": "GPD",
            "expected_basis": daily.get("default_target_basis"),
            "expected_source": "well_chem_program",
            "expected_source_priority": daily.get("target_source_preference"),
            "expected_confidence": "MEDIUM",
            "expected_status": "ACTIVE",
        }
    ).drop_duplicates()

    write_table(expected, modeled_dir, "fact_expected_chem_daily", settings)
    batch.set_row_count("fact_expected_chem_daily", len(expected))
    logger.info(
        "Built fact_expected_chem_daily from well_chem_program | rows=%s",
        len(expected),
    )
    return expected