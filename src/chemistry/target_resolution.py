import pandas as pd

from src.chemistry.chemical_mapping import load_basis_rules, map_chemical_names
from src.common.constants import CONFIDENCE_HIGH
from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


def build_fact_chem_target_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    rates_path = staged_dir / "stg_chemical_rates.csv"
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    if not rates_path.exists():
        logger.warning("stg_chemical_rates.csv not found.")
        return pd.DataFrame()

    rates = pd.read_csv(rates_path, dtype={"well_id": str})
    rates["date"] = pd.to_datetime(rates["date"], errors="coerce")

    mapped, exceptions = map_chemical_names(
        rates,
        chem_name_col="chem_name",
        chem_type_col="chem_type",
        table_name="stg_chemical_rates",
        batch_id=batch.batch_id,
    )

    if exceptions:
        append_exceptions(exceptions)

    # fallback mapping using dim_chemical
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

    basis_rules = load_basis_rules()
    if not basis_rules.empty:
        mapped = mapped.merge(basis_rules, how="left", on="chemical_key")

    fact = pd.DataFrame(
        {
            "well_id": mapped["well_id"],
            "date": mapped["date"].dt.date,
            "chemical_key": mapped["chemical_key"],
            "target_rate": pd.to_numeric(mapped["target_gpd"], errors="coerce"),
            "target_unit": "GPD",
            "target_basis": mapped["dose_basis"],
            "target_source": "chemical_rates",
            "target_source_priority": 3,
            "effective_rule_used": "DAILY_RATES_FILE",
            "target_grain": "DAILY",
            "target_confidence": CONFIDENCE_HIGH,
            "target_status": "ACTIVE",
        }
    )

    if "default_basis" in mapped.columns:
        fact["target_basis"] = fact["target_basis"].where(
            fact["target_basis"].notna(),
            mapped["default_basis"],
        )

    missing_key = fact["chemical_key"].isna().sum()
    if missing_key > 0:
        logger.warning("Dropping chem target rows with missing chemical_key | rows=%s", missing_key)

    fact = fact[fact["chemical_key"].notna()].copy()
    fact = fact.drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")

    write_table(fact, modeled_dir, "fact_chem_target_daily", settings)
    batch.set_row_count("fact_chem_target_daily", len(fact))
    logger.info("Built fact_chem_target_daily | rows=%s", len(fact))
    return fact