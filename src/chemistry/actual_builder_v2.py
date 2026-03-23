from __future__ import annotations

import re
import pandas as pd

from src.chemistry.chemical_mapping import map_chemical_names
from src.common.constants import ACTUAL_METHOD_UNKNOWN, CONFIDENCE_UNKNOWN
from src.common.paths import FACT_PRODUCTION, get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


# -----------------------------
# Helpers
# -----------------------------
def _normalize_chem_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    text = re.sub(r"[\s\-/]+", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _normalize_type_text(value) -> str | None:
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# -----------------------------
# Main Builder
# -----------------------------
def build_fact_chem_actual_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    cost_path = staged_dir / "stg_chemical_cost.csv"
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    if not cost_path.exists():
        logger.warning("stg_chemical_cost.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(cost_path, dtype={"well_id": str})

    # -----------------------------
    # Basic cleanup
    # -----------------------------
    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["qty"] = _safe_numeric(df.get("qty"))
    df["actual_cost"] = _safe_numeric(df.get("actual_cost"))

    df["line_category"] = df["line_category"].astype("string").str.upper()

    # -----------------------------
    # SPLIT DATA
    # -----------------------------
    chem_df = df[df["line_category"] == "CHEMICAL"].copy()
    equip_df = df[df["line_category"] == "EQUIPMENT"].copy()
    disc_df = df[df["line_category"] == "DISCOUNT"].copy()

    # -----------------------------
    # CHEMICALS → mapping
    # -----------------------------
    if not chem_df.empty:
        mapped, exceptions = map_chemical_names(
            chem_df,
            chem_name_col="chem_name",
            chem_type_col="chem_type",
            table_name="stg_chemical_cost",
            batch_id=batch.batch_id,
        )

        if exceptions:
            append_exceptions(exceptions)

        if "chemical_key" not in mapped.columns:
            mapped["chemical_key"] = pd.NA

        mapped["chem_name_norm"] = mapped["chem_name"].apply(_normalize_chem_text)
        mapped["chem_type_norm"] = mapped["chem_type"].apply(_normalize_type_text)

    else:
        mapped = pd.DataFrame(columns=["chemical_key"])

    # -----------------------------
    # fallback mapping
    # -----------------------------
    if not mapped.empty and chem_dim_path.exists():
        chem_dim = pd.read_csv(chem_dim_path)

        chem_dim["chem_name_norm"] = chem_dim["normalized_chemical_name"].apply(_normalize_chem_text)
        chem_dim["chem_type_norm"] = chem_dim["chem_type"].apply(_normalize_type_text)

        merged = mapped.merge(
            chem_dim[["chemical_key", "chem_name_norm", "chem_type_norm"]],
            how="left",
            on=["chem_name_norm", "chem_type_norm"],
            suffixes=("", "_dim"),
        )

        if "chemical_key_dim" in merged.columns:
            merged["chemical_key"] = merged["chemical_key"].where(
                merged["chemical_key"].notna(),
                merged["chemical_key_dim"],
            )
            merged = merged.drop(columns=["chemical_key_dim"])

        mapped = merged

    # -----------------------------
    # BUILD CHEM FACT (DAILY)
    # -----------------------------
    chem_fact = pd.DataFrame()

    if not mapped.empty:
        chem_fact = pd.DataFrame(
            {
                "well_id": mapped["well_id"],
                "chemical_key": mapped["chemical_key"],
                "date": mapped["date"].dt.date,
                "actual_gpd": mapped["qty"],
                "actual_cost": mapped["actual_cost"],
                "actual_unit": mapped.get("uom"),
                "actual_method": ACTUAL_METHOD_UNKNOWN,
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "equipment": mapped.get("equipment"),
                "vendor": mapped.get("vendor"),
                "line_category": "CHEMICAL",
            }
        )

    # -----------------------------
    # AGGREGATE DAILY
    # -----------------------------
    if not chem_fact.empty:
        chem_fact = (
            chem_fact.groupby(["well_id", "chemical_key", "date"], as_index=False)
            .agg(
                {
                    "actual_gpd": "sum",
                    "actual_cost": "sum",
                    "actual_unit": "first",
                    "actual_method": "first",
                    "actual_confidence": "first",
                }
            )
        )

    # -----------------------------
    # PRODUCTION JOIN (for ppm)
    # -----------------------------
    prod = pd.read_csv(FACT_PRODUCTION, dtype={"well_id": str})
    prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.date

    if "oil_bbl" in prod.columns and "bopm" not in prod.columns:
        prod = prod.rename(columns={"oil_bbl": "bopm"})
    if "water_bbl" in prod.columns and "bwpm" not in prod.columns:
        prod = prod.rename(columns={"water_bbl": "bwpm"})

    prod["bopm"] = _safe_numeric(prod["bopm"])
    prod["bwpm"] = _safe_numeric(prod["bwpm"])

    prod["total_liquid_bbl"] = prod["bopm"].fillna(0) + prod["bwpm"].fillna(0)

    if not chem_fact.empty:
        chem_fact = chem_fact.merge(
            prod[["well_id", "date", "bopm", "bwpm", "total_liquid_bbl"]],
            on=["well_id", "date"],
            how="left",
        )

        chem_fact["actual_ppm"] = (
            chem_fact["actual_gpd"] * 1_000_000
        ) / (chem_fact["total_liquid_bbl"] * 42)

    # -----------------------------
    # FLAGS (ENGINEERING)
    # -----------------------------
    if not chem_fact.empty:
        chem_fact["actual_has_data_flag"] = chem_fact["actual_gpd"].notna().astype(int)

        chem_fact["no_chemical_flag"] = (
            chem_fact["actual_gpd"].fillna(0) <= 0
        ).astype(int)

        chem_fact["missing_actual_flag"] = chem_fact["actual_gpd"].isna().astype(int)

        chem_fact["actual_calc_valid_flag"] = (
            chem_fact["actual_gpd"].notna()
        ).astype(int)

    # -----------------------------
    # FINAL CLEAN
    # -----------------------------
    fact = chem_fact.copy()

    fact = fact.drop_duplicates(
        subset=["well_id", "date", "chemical_key"],
        keep="last",
    )

    # -----------------------------
    # WRITE
    # -----------------------------
    write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
    batch.set_row_count("fact_chem_actual_daily", len(fact))
    logger.info("Built fact_chem_actual_daily | rows=%s", len(fact))

    return fact