from __future__ import annotations

import re
import pandas as pd

from src.chemistry.chemical_mapping import map_chemical_names
from src.common.constants import ACTUAL_METHOD_UNKNOWN, CONFIDENCE_UNKNOWN
from src.common.paths import get_path
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

    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df["actual_cost"] = pd.to_numeric(df.get("actual_cost"), errors="coerce")

    df["line_category"] = df["line_category"].astype("string").str.upper()

    # -----------------------------
    # SPLIT DATA (THIS IS THE FIX)
    # -----------------------------
    chem_df = df[df["line_category"] == "CHEMICAL"].copy()
    equip_df = df[df["line_category"] == "EQUIPMENT"].copy()
    disc_df = df[df["line_category"] == "DISCOUNT"].copy()

    # -----------------------------
    # CHEMICALS → mapping required
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
            logger.warning("chemical_key not returned from mapping. Creating empty column.")
            mapped["chemical_key"] = pd.NA

        mapped["chem_name_norm"] = mapped["chem_name"].apply(_normalize_chem_text)
        mapped["chem_type_norm"] = mapped["chem_type"].apply(_normalize_type_text)

    else:
        mapped = pd.DataFrame(columns=["chemical_key"])

    # -----------------------------
    # OPTIONAL: fallback mapping
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

        # -----------------------------
        # FIX: unify chemical_key
        # -----------------------------
        if "chemical_key_dim" in merged.columns:
            if "chemical_key" in merged.columns:
                merged["chemical_key"] = merged["chemical_key"].where(
                    merged["chemical_key"].notna(),
                    merged["chemical_key_dim"],
                )
            else:
                merged["chemical_key"] = merged["chemical_key_dim"]

            merged = merged.drop(columns=["chemical_key_dim"])

        mapped = merged

    # -----------------------------
    # BUILD CHEMICAL FACT
    # -----------------------------
    chem_fact = pd.DataFrame()

    if not mapped.empty:
        chem_fact = pd.DataFrame(
            {
                "well_id": mapped["well_id"],
                "chemical_key": mapped["chemical_key"],
                "period_start": mapped["date"].dt.date,
                "period_end": mapped["date"].dt.date,
                "actual_total_volume": mapped["qty"],
                "actual_total_cost": mapped["actual_cost"],
                "actual_unit": mapped.get("uom"),
                "source": "chemical_cost",
                "allocation_method": "NONE",
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "actual_method": ACTUAL_METHOD_UNKNOWN,
                "equipment": mapped.get("equipment"),
                "vendor": mapped.get("vendor"),
                "line_category": "CHEMICAL",
                "chem_name_raw": mapped["chem_name"],
                "chem_type_raw": mapped["chem_type"],
            }
        )

    # -----------------------------
    # EQUIPMENT FACT (NO MAPPING)
    # -----------------------------
    equip_fact = pd.DataFrame()

    if not equip_df.empty:
        equip_fact = pd.DataFrame(
            {
                "well_id": equip_df["well_id"],
                "chemical_key": None,
                "period_start": equip_df["date"].dt.date,
                "period_end": equip_df["date"].dt.date,
                "actual_total_volume": None,
                "actual_total_cost": equip_df["actual_cost"],
                "actual_unit": equip_df.get("uom"),
                "source": "chemical_cost",
                "allocation_method": "NONE",
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "actual_method": "EQUIPMENT",
                "equipment": equip_df["equipment"],
                "vendor": equip_df.get("vendor"),
                "line_category": "EQUIPMENT",
                "chem_name_raw": None,
                "chem_type_raw": None,
            }
        )

    # -----------------------------
    # DISCOUNT FACT
    # -----------------------------
    disc_fact = pd.DataFrame()

    if not disc_df.empty:
        disc_fact = pd.DataFrame(
            {
                "well_id": disc_df["well_id"],
                "chemical_key": None,
                "period_start": disc_df["date"].dt.date,
                "period_end": disc_df["date"].dt.date,
                "actual_total_volume": None,
                "actual_total_cost": disc_df["actual_cost"],
                "actual_unit": None,
                "source": "chemical_cost",
                "allocation_method": "NONE",
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "actual_method": "DISCOUNT",
                "equipment": None,
                "vendor": disc_df.get("vendor"),
                "line_category": "DISCOUNT",
                "chem_name_raw": None,
                "chem_type_raw": None,
            }
        )

    # -----------------------------
    # COMBINE ALL
    # -----------------------------
    frames = [chem_fact, equip_fact, disc_fact]

    frames = [f for f in frames if f is not None and not f.empty and not f.isna().all().all()]

    fact = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # -----------------------------
    # CLEANUP
    # -----------------------------
    fact = fact[fact["period_start"].notna()].copy()

    fact = fact.drop_duplicates(
        subset=["well_id", "period_start", "chemical_key", "line_category"],
        keep="last",
    )

    # -----------------------------
    # WRITE OUTPUT
    # -----------------------------
    write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
    batch.set_row_count("fact_chem_actual_daily", len(fact))
    logger.info("Built fact_chem_actual_daily | rows=%s", len(fact))

    return fact