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
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    if not cost_path.exists():
        logger.warning("stg_chemical_cost.csv not found.")
        return pd.DataFrame()

    # -----------------------------
    # Load staged data
    # -----------------------------
    cost = pd.read_csv(cost_path, dtype={"well_id": str})

    if "date" in cost.columns:
        cost["date"] = pd.to_datetime(cost["date"], errors="coerce")

    # -----------------------------
    # Primary mapping (existing logic)
    # -----------------------------
    mapped, exceptions = map_chemical_names(
        cost,
        chem_name_col="chem_name",
        chem_type_col="chem_type",
        table_name="stg_chemical_cost",
        batch_id=batch.batch_id,
    )

    if exceptions:
        append_exceptions(exceptions)

    # -----------------------------
    # Fallback mapping (THIS FIXES YOUR ISSUE)
    # -----------------------------
    if chem_dim_path.exists():
        chem_dim = pd.read_csv(chem_dim_path, dtype={"chemical_key": str})

        chem_dim["name_key"] = (
            chem_dim["normalized_chemical_name"].astype(str).str.strip().str.lower()
        )
        chem_dim["type_key"] = chem_dim["chem_type"].astype(str).str.strip().str.lower()

        mapped["name_key"] = mapped["chem_name"].astype(str).str.strip().str.lower()
        mapped["type_key"] = mapped["chem_type"].astype(str).str.strip().str.lower()

        fallback = mapped.merge(
            chem_dim[["chemical_key", "name_key", "type_key"]],
            how="left",
            on=["name_key", "type_key"],
            suffixes=("", "_fallback"),
        )

        if "chemical_key_fallback" in fallback.columns:
            fallback["chemical_key"] = fallback["chemical_key"].where(
                fallback["chemical_key"].notna(),
                fallback["chemical_key_fallback"],
            )
            fallback = fallback.drop(columns=["chemical_key_fallback"])

        mapped = fallback.drop(columns=["name_key", "type_key"])

    # -----------------------------
    # Build fact table
    # -----------------------------
    fact = pd.DataFrame(
        {
            "well_id": mapped["well_id"],
            "chemical_key": mapped["chemical_key"],
            "period_start": mapped["date"].dt.date,
            "period_end": mapped["date"].dt.date,
            "actual_total_volume": pd.to_numeric(mapped["qty"], errors="coerce"),
            "actual_total_cost": pd.to_numeric(mapped["actual_cost"], errors="coerce"),
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

    # -----------------------------
    # Drop bad rows (THIS WAS YOUR 0-ROW ISSUE)
    # -----------------------------
    missing_key = fact["chemical_key"].isna().sum()
    if missing_key > 0:
        logger.warning(
            "Dropping chem actual rows with missing chemical_key | rows=%s",
            missing_key,
        )

    fact = fact[fact["chemical_key"].notna()].copy()

    # Ensure correct grain
    fact = fact.drop_duplicates(
        subset=["well_id", "period_start", "chemical_key"],
        keep="last",
    )

    # -----------------------------
    # Write output
    # -----------------------------
    write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
    batch.set_row_count("fact_chem_actual_daily", len(fact))
    logger.info("Built fact_chem_actual_daily | rows=%s", len(fact))

    return fact