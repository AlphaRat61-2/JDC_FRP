from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.chemistry.ppm_calculations import add_target_ppm
from src.io.writers import write_table
from src.reconciliation.compliance_status import (
    classify_operational_status,
    load_chem_tolerance,
)
from src.reconciliation.exception_rules import assign_exception_code
from src.reconciliation.financial_status import classify_financial_status


def _dedupe_expected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    keep = [
        "well_id",
        "date",
        "chemical_key",
        "expected_rate",
        "expected_unit",
        "expected_basis",
        "expected_source",
        "expected_source_priority",
        "expected_confidence",
        "expected_status",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    return out


def _dedupe_target(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    keep = [
        "well_id",
        "date",
        "chemical_key",
        "target_rate",
        "target_unit",
        "target_basis",
        "target_source",
        "target_source_priority",
        "target_confidence",
        "target_status",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    out["target_exists_flag"] = out["target_rate"].notna()
    return out


def _dedupe_actual(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    elif "period_start" in out.columns:
        out["date"] = pd.to_datetime(out["period_start"], errors="coerce").dt.date
    else:
        return pd.DataFrame(columns=["well_id", "date", "chemical_key"])

    agg_map = {}
    if "actual_rate" in out.columns:
        agg_map["actual_rate"] = "sum"
    if "actual_total_cost" in out.columns:
        agg_map["spend"] = ("actual_total_cost", "sum")
    elif "spend" in out.columns:
        agg_map["spend"] = "sum"

    if not agg_map:
        out = out[["well_id", "date", "chemical_key"]].drop_duplicates()
        out["actual_exists_flag"] = False
        out["spend_exists_flag"] = False
        return out

    grouped = (
        out.groupby(["well_id", "date", "chemical_key"], as_index=False)
        .agg(**{k: v for k, v in agg_map.items() if isinstance(v, tuple)})
        if any(isinstance(v, tuple) for v in agg_map.values())
        else out.groupby(["well_id", "date", "chemical_key"], as_index=False).agg(agg_map)
    )

    if "actual_rate" in grouped.columns:
        grouped["actual_exists_flag"] = grouped["actual_rate"].notna()
    else:
        grouped["actual_rate"] = pd.NA
        grouped["actual_exists_flag"] = False

    if "spend" in grouped.columns:
        grouped["spend_exists_flag"] = grouped["spend"].notna()
    else:
        grouped["spend"] = pd.NA
        grouped["spend_exists_flag"] = False

    return grouped


def build_fact_chem_recon_daily(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")

    expected_path = modeled_dir / "fact_expected_chem_daily.csv"
    target_path = modeled_dir / "fact_chem_target_daily.csv"
    prod_path = modeled_dir / "fact_production_daily.csv"
    actual_path = modeled_dir / "fact_chem_actual_daily.csv"
    chem_dim_path = modeled_dir / "dim_chemical.csv"

    expected = (
        pd.read_csv(expected_path, dtype={"well_id": str, "chemical_key": str})
        if expected_path.exists()
        else pd.DataFrame()
    )
    target = (
        pd.read_csv(target_path, dtype={"well_id": str, "chemical_key": str})
        if target_path.exists()
        else pd.DataFrame()
    )
    prod = (
        pd.read_csv(prod_path, dtype={"well_id": str})
        if prod_path.exists()
        else pd.DataFrame()
    )
    actual = (
        pd.read_csv(actual_path, dtype={"well_id": str, "chemical_key": str})
        if actual_path.exists()
        else pd.DataFrame()
    )
    chem_dim = (
        pd.read_csv(chem_dim_path, dtype={"chemical_key": str})
        if chem_dim_path.exists()
        else pd.DataFrame()
    )

    if expected.empty and target.empty and actual.empty:
        logger.warning("Skipping chemical reconciliation — no chemistry inputs found.")
        empty = pd.DataFrame()
        write_table(empty, modeled_dir, "fact_chem_recon_daily", settings)
        batch.set_row_count("fact_chem_recon_daily", 0)
        logger.info("Built fact_chem_recon_daily | rows=0")
        return empty

    merge_keys = ["well_id", "date", "chemical_key"]

    expected = _dedupe_expected(expected)
    target = _dedupe_target(target)
    actual = _dedupe_actual(actual)

    if not prod.empty and "date" in prod.columns:
        prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.date
        prod = (
            prod[["well_id", "date", "oil_bbl", "water_bbl", "gas_mcf"]]
            .drop_duplicates(subset=["well_id", "date"], keep="last")
        )

    if not chem_dim.empty:
        chem_dim = chem_dim[["chemical_key", "chem_type", "dose_basis"]].drop_duplicates(
            subset=["chemical_key"], keep="last"
        )

    # canonical daily target/expected grain
    if expected.empty:
        recon = target.copy()
        recon["expected_flag"] = False
    else:
        recon = expected.merge(target, how="outer", on=merge_keys, suffixes=("_exp", ""))
        if "expected_status" in recon.columns:
            recon["expected_flag"] = recon["expected_status"].fillna("ACTIVE").eq("ACTIVE")
        elif "expected_rate" in recon.columns:
            recon["expected_flag"] = recon["expected_rate"].notna()
        else:
            recon["expected_flag"] = False

    # align expected -> target fields when target missing
    if "target_rate" not in recon.columns:
        recon["target_rate"] = pd.NA
    if "expected_rate" in recon.columns:
        recon["target_rate"] = recon["target_rate"].where(recon["target_rate"].notna(), recon["expected_rate"])

    if "target_basis" not in recon.columns:
        recon["target_basis"] = pd.NA
    if "expected_basis" in recon.columns:
        recon["target_basis"] = recon["target_basis"].where(recon["target_basis"].notna(), recon["expected_basis"])

    if "target_exists_flag" not in recon.columns:
        recon["target_exists_flag"] = recon["target_rate"].notna()

    recon = recon.drop_duplicates(subset=merge_keys, keep="last")

    if not chem_dim.empty:
        recon = recon.merge(chem_dim, how="left", on="chemical_key")
        recon["target_basis"] = recon["target_basis"].where(recon["target_basis"].notna(), recon["dose_basis"])

    if not prod.empty:
        recon = recon.merge(prod, how="left", on=["well_id", "date"])

    if not actual.empty:
        recon = recon.merge(actual, how="left", on=merge_keys)
    else:
        recon["actual_rate"] = pd.NA
        recon["actual_exists_flag"] = False
        recon["spend"] = pd.NA
        recon["spend_exists_flag"] = False

    recon = add_target_ppm(recon)

    tol_df = load_chem_tolerance()

    recon["operational_status"] = recon.apply(
        lambda r: classify_operational_status(
            target_rate=r.get("target_rate"),
            actual_rate=r.get("actual_rate"),
            chem_type=r.get("chem_type"),
            tol_df=tol_df,
        ),
        axis=1,
    )

    recon["financial_status"] = recon.apply(
        lambda r: classify_financial_status(
            target_exists=bool(r.get("target_exists_flag")),
            spend_exists=bool(r.get("spend_exists_flag")),
        ),
        axis=1,
    )

    recon["exception_code"] = recon.apply(assign_exception_code, axis=1)

    keep_cols = [
        "well_id",
        "date",
        "chemical_key",
        "chem_type",
        "expected_flag",
        "target_exists_flag",
        "actual_exists_flag",
        "spend_exists_flag",
        "target_rate",
        "actual_rate",
        "target_basis",
        "target_ppm",
        "actual_ppm",
        "spend",
        "variance_rate_pct",
        "variance_ppm_pct",
        "operational_status",
        "financial_status",
        "exception_code",
        "confidence_status",
    ]
    keep_cols = [c for c in keep_cols if c in recon.columns]
    recon = recon[keep_cols].drop_duplicates(subset=merge_keys, keep="last").copy()

    write_table(recon, modeled_dir, "fact_chem_recon_daily", settings)
    batch.set_row_count("fact_chem_recon_daily", len(recon))
    logger.info("Built fact_chem_recon_daily | rows=%s", len(recon))
    return recon