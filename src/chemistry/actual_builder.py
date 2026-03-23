from __future__ import annotations

import re
import pandas as pd

from src.chemistry.chemical_mapping import map_chemical_names
from src.common.constants import ACTUAL_METHOD_UNKNOWN, CONFIDENCE_UNKNOWN
from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


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


def build_fact_chem_actual_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    cost_path = staged_dir / "stg_chemical_cost.csv"
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    if not cost_path.exists():
        logger.warning("stg_chemical_cost.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(cost_path, dtype={"well_id": str})

    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df["actual_cost"] = pd.to_numeric(df.get("actual_cost"), errors="coerce")
    df["line_category"] = df["line_category"].astype("string").str.upper()

    chem_df = df[df["line_category"] == "CHEMICAL"].copy()
    equip_df = df[df["line_category"] == "EQUIPMENT"].copy()
    disc_df = df[df["line_category"] == "DISCOUNT"].copy()

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
            if "chemical_key" in merged.columns:
                merged["chemical_key"] = merged["chemical_key"].where(
                    merged["chemical_key"].notna(),
                    merged["chemical_key_dim"],
                )
            else:
                merged["chemical_key"] = merged["chemical_key_dim"]

            merged = merged.drop(columns=["chemical_key_dim"])

        mapped = merged

    chem_fact = pd.DataFrame()

    if not mapped.empty:
        chem_fact = pd.DataFrame(
            {
                "well_id": mapped["well_id"].astype(str).str.strip(),
                "chemical_key": mapped["chemical_key"],
                "period_start": pd.to_datetime(mapped["date"], errors="coerce").dt.date,
                "period_end": pd.to_datetime(mapped["date"], errors="coerce").dt.date,
                "actual_total_volume": pd.to_numeric(mapped["qty"], errors="coerce"),
                "actual_total_cost": pd.to_numeric(mapped["actual_cost"], errors="coerce"),
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

    equip_fact = pd.DataFrame()

    if not equip_df.empty:
        equip_fact = pd.DataFrame(
            {
                "well_id": equip_df["well_id"].astype(str).str.strip(),
                "chemical_key": pd.NA,
                "period_start": pd.to_datetime(equip_df["date"], errors="coerce").dt.date,
                "period_end": pd.to_datetime(equip_df["date"], errors="coerce").dt.date,
                "actual_total_volume": pd.NA,
                "actual_total_cost": pd.to_numeric(equip_df["actual_cost"], errors="coerce"),
                "actual_unit": equip_df.get("uom"),
                "source": "chemical_cost",
                "allocation_method": "NONE",
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "actual_method": "EQUIPMENT",
                "equipment": equip_df.get("equipment"),
                "vendor": equip_df.get("vendor"),
                "line_category": "EQUIPMENT",
                "chem_name_raw": pd.NA,
                "chem_type_raw": pd.NA,
            }
        )

    disc_fact = pd.DataFrame()

    if not disc_df.empty:
        disc_fact = pd.DataFrame(
            {
                "well_id": disc_df["well_id"].astype(str).str.strip(),
                "chemical_key": pd.NA,
                "period_start": pd.to_datetime(disc_df["date"], errors="coerce").dt.date,
                "period_end": pd.to_datetime(disc_df["date"], errors="coerce").dt.date,
                "actual_total_volume": pd.NA,
                "actual_total_cost": pd.to_numeric(disc_df["actual_cost"], errors="coerce"),
                "actual_unit": pd.NA,
                "source": "chemical_cost",
                "allocation_method": "NONE",
                "actual_confidence": CONFIDENCE_UNKNOWN,
                "actual_method": "DISCOUNT",
                "equipment": pd.NA,
                "vendor": disc_df.get("vendor"),
                "line_category": "DISCOUNT",
                "chem_name_raw": pd.NA,
                "chem_type_raw": pd.NA,
            }
        )

    clean_frames = []
    for f in [chem_fact, equip_fact, disc_fact]:
        if f is None or f.empty:
            continue

        f = f.copy()
        f = f.loc[:, ~f.isna().all()]
        if f.empty:
            continue

        if "chemical_key" not in f.columns:
            f["chemical_key"] = pd.NA

        clean_frames.append(f)

    fact = pd.concat(clean_frames, ignore_index=True, sort=False) if clean_frames else pd.DataFrame()

    if fact.empty:
        write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
        batch.set_row_count("fact_chem_actual_daily", 0)
        logger.info("Built fact_chem_actual_daily | rows=0")
        return fact

    if "chemical_key" not in fact.columns:
        fact["chemical_key"] = pd.NA

    fact = fact[fact["period_start"].notna()].copy()

    fact = fact.drop_duplicates(
        subset=["well_id", "period_start", "chemical_key", "line_category"],
        keep="last",
    )

    write_table(fact, modeled_dir, "fact_chem_actual_daily", settings)
    batch.set_row_count("fact_chem_actual_daily", len(fact))
    logger.info("Built fact_chem_actual_daily | rows=%s", len(fact))

    return fact