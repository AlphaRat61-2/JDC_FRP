from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table
from src.reconciliation.compliance_status import (
    classify_operational_status,
    load_chem_tolerance,
)
from src.reconciliation.exception_rules import assign_exception_code
from src.reconciliation.financial_status import classify_financial_status


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _dedupe_expected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    keep = [
        "well_id",
        "date",
        "chemical_key",
        "expected_gpd",
        "expected_ppm",
        "expected_basis",
        "expected_source",
        "expected_source_priority",
        "expected_confidence",
        "expected_status",
        "production_basis_volume_bbl",
        "bopm",
        "bwpm",
        "missing_production_flag",
        "missing_target_flag",
        "no_production_flag",
        "stale_target_flag",
        "expected_calc_valid_flag",
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
        "chem_name",
        "chem_type",
        "target_gpd",
        "target_ppm",
        "target_basis",
        "target_active_flag",
        "target_effective_date",
        "target_last_update_date",
        "target_age_days",
        "stale_target_flag",
        "bopm",
        "bwpm",
        "total_liquid_bbl",
    ]
    keep = [c for c in keep if c in out.columns]

    out = out[keep].drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    out["target_exists_flag"] = out["target_gpd"].notna()
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

    agg_dict = {}
    if "actual_gpd" in out.columns:
        agg_dict["actual_gpd"] = "sum"
    if "actual_cost" in out.columns:
        agg_dict["actual_cost"] = "sum"
    if "actual_ppm" in out.columns:
        agg_dict["actual_ppm"] = "mean"
    if "actual_has_data_flag" in out.columns:
        agg_dict["actual_has_data_flag"] = "max"
    if "no_chemical_flag" in out.columns:
        agg_dict["no_chemical_flag"] = "max"
    if "missing_actual_flag" in out.columns:
        agg_dict["missing_actual_flag"] = "max"
    if "actual_calc_valid_flag" in out.columns:
        agg_dict["actual_calc_valid_flag"] = "max"

    if not agg_dict:
        out = out[["well_id", "date", "chemical_key"]].drop_duplicates()
        out["actual_gpd"] = pd.NA
        out["actual_cost"] = pd.NA
        out["actual_ppm"] = pd.NA
        out["actual_exists_flag"] = False
        out["spend_exists_flag"] = False
        out["actual_has_data_flag"] = 0
        out["no_chemical_flag"] = 0
        out["missing_actual_flag"] = 1
        out["actual_calc_valid_flag"] = 0
        return out

    grouped = (
        out.groupby(["well_id", "date", "chemical_key"], as_index=False)
        .agg(agg_dict)
    )

    if "actual_gpd" not in grouped.columns:
        grouped["actual_gpd"] = pd.NA
    if "actual_cost" not in grouped.columns:
        grouped["actual_cost"] = pd.NA
    if "actual_ppm" not in grouped.columns:
        grouped["actual_ppm"] = pd.NA
    if "actual_has_data_flag" not in grouped.columns:
        grouped["actual_has_data_flag"] = grouped["actual_gpd"].notna().astype(int)
    if "no_chemical_flag" not in grouped.columns:
        grouped["no_chemical_flag"] = (grouped["actual_gpd"].fillna(0) <= 0).astype(int)
    if "missing_actual_flag" not in grouped.columns:
        grouped["missing_actual_flag"] = grouped["actual_gpd"].isna().astype(int)
    if "actual_calc_valid_flag" not in grouped.columns:
        grouped["actual_calc_valid_flag"] = grouped["actual_gpd"].notna().astype(int)

    grouped["actual_exists_flag"] = grouped["actual_gpd"].notna()
    grouped["spend_exists_flag"] = grouped["actual_cost"].notna()

    return grouped


def _calc_pct_variance(actual, base):
    actual = pd.to_numeric(actual, errors="coerce")
    base = pd.to_numeric(base, errors="coerce")

    if pd.isna(actual) or pd.isna(base) or base == 0:
        return pd.NA

    return (actual - base) / base


def _calc_abs_variance(actual, base):
    actual = pd.to_numeric(actual, errors="coerce")
    base = pd.to_numeric(base, errors="coerce")

    if pd.isna(actual) or pd.isna(base):
        return pd.NA

    return actual - base


def _classify_feed_status(row, tol_df: pd.DataFrame) -> str:
    if int(row.get("missing_target_flag", 0)) == 1:
        return "MISSING_TARGET"
    if int(row.get("missing_actual_flag", 0)) == 1:
        return "MISSING_ACTUAL"
    if int(row.get("no_production_flag", 0)) == 1:
        return "NO_PRODUCTION"
    if int(row.get("no_chemical_flag", 0)) == 1:
        return "NO_CHEMICAL"
    if int(row.get("stale_target_flag", 0)) == 1:
        return "STALE_TARGET"

    expected_gpd = pd.to_numeric(row.get("expected_gpd"), errors="coerce")
    actual_gpd = pd.to_numeric(row.get("actual_gpd"), errors="coerce")

    if pd.isna(expected_gpd) or pd.isna(actual_gpd) or expected_gpd == 0:
        return "UNKNOWN"

    tol = 0.10
    chem_type = row.get("chem_type")

    if tol_df is not None and not tol_df.empty and "chem_type" in tol_df.columns:
        match = tol_df[tol_df["chem_type"].astype(str) == str(chem_type)]
        if not match.empty:
            if "tolerance_pct" in match.columns:
                tol = pd.to_numeric(match.iloc[0]["tolerance_pct"], errors="coerce")
            elif "tolerance" in match.columns:
                tol = pd.to_numeric(match.iloc[0]["tolerance"], errors="coerce")
            tol = 0.10 if pd.isna(tol) else float(tol)

    var_pct = _calc_pct_variance(actual_gpd, expected_gpd)

    if pd.isna(var_pct):
        return "UNKNOWN"
    if var_pct < -tol:
        return "UNDERFEED"
    if var_pct > tol:
        return "OVERFEED"
    return "ON_TARGET"


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

        rename_map = {}
        if "oil_bbl" in prod.columns and "bopm" not in prod.columns:
            rename_map["oil_bbl"] = "bopm"
        if "water_bbl" in prod.columns and "bwpm" not in prod.columns:
            rename_map["water_bbl"] = "bwpm"
        if rename_map:
            prod = prod.rename(columns=rename_map)

        prod_keep = ["well_id", "date"]
        for c in ["bopm", "bwpm", "gas_mcf"]:
            if c in prod.columns:
                prod_keep.append(c)

        prod = prod[prod_keep].drop_duplicates(subset=["well_id", "date"], keep="last")

        if "bopm" in prod.columns:
            prod["bopm"] = _safe_numeric(prod["bopm"])
        else:
            prod["bopm"] = pd.NA

        if "bwpm" in prod.columns:
            prod["bwpm"] = _safe_numeric(prod["bwpm"])
        else:
            prod["bwpm"] = pd.NA

        prod["production_basis_volume_bbl"] = prod["bopm"].fillna(0) + prod["bwpm"].fillna(0)

    if not chem_dim.empty:
        keep = [c for c in ["chemical_key", "chem_type", "dose_basis"] if c in chem_dim.columns]
        chem_dim = chem_dim[keep].drop_duplicates(subset=["chemical_key"], keep="last")

    if expected.empty:
        recon = target.copy()
        recon["expected_flag"] = False
    else:
        recon = expected.merge(target, how="outer", on=merge_keys, suffixes=("_exp", ""))

        if "expected_status" in recon.columns:
            recon["expected_flag"] = recon["expected_status"].fillna("ACTIVE").eq("ACTIVE")
        elif "expected_gpd" in recon.columns:
            recon["expected_flag"] = recon["expected_gpd"].notna()
        else:
            recon["expected_flag"] = False

    if "target_gpd" not in recon.columns:
        recon["target_gpd"] = pd.NA
    if "expected_gpd" in recon.columns:
        recon["target_gpd"] = recon["target_gpd"].where(recon["target_gpd"].notna(), recon["expected_gpd"])

    if "target_ppm" not in recon.columns:
        recon["target_ppm"] = pd.NA
    if "expected_ppm" in recon.columns:
        recon["target_ppm"] = recon["target_ppm"].where(recon["target_ppm"].notna(), recon["expected_ppm"])

    if "target_basis" not in recon.columns:
        recon["target_basis"] = pd.NA
    if "expected_basis" in recon.columns:
        recon["target_basis"] = recon["target_basis"].where(recon["target_basis"].notna(), recon["expected_basis"])

    if "target_exists_flag" not in recon.columns:
        recon["target_exists_flag"] = recon["target_gpd"].notna()

    recon = recon.drop_duplicates(subset=merge_keys, keep="last")

    if not chem_dim.empty:
        recon = recon.merge(chem_dim, how="left", on="chemical_key")
        if "dose_basis" in recon.columns:
            recon["target_basis"] = recon["target_basis"].where(recon["target_basis"].notna(), recon["dose_basis"])

        if "chem_type_x" in recon.columns and "chem_type_y" in recon.columns:
            recon["chem_type"] = recon["chem_type_x"].where(recon["chem_type_x"].notna(), recon["chem_type_y"])
            recon = recon.drop(columns=["chem_type_x", "chem_type_y"])
        elif "chem_type_x" in recon.columns:
            recon = recon.rename(columns={"chem_type_x": "chem_type"})
        elif "chem_type_y" in recon.columns:
            recon = recon.rename(columns={"chem_type_y": "chem_type"})

    if not prod.empty:
        recon = recon.merge(prod, how="left", on=["well_id", "date"])

    if not actual.empty:
        recon = recon.merge(actual, how="left", on=merge_keys)
    else:
        recon["actual_gpd"] = pd.NA
        recon["actual_cost"] = pd.NA
        recon["actual_ppm"] = pd.NA
        recon["actual_exists_flag"] = False
        recon["spend_exists_flag"] = False
        recon["actual_has_data_flag"] = 0
        recon["no_chemical_flag"] = 0
        recon["missing_actual_flag"] = 1
        recon["actual_calc_valid_flag"] = 0

    for col in [
        "target_gpd",
        "expected_gpd",
        "actual_gpd",
        "target_ppm",
        "expected_ppm",
        "actual_ppm",
        "actual_cost",
        "bopm",
        "bwpm",
        "production_basis_volume_bbl",
    ]:
        if col in recon.columns:
            recon[col] = _safe_numeric(recon[col])

    if "production_basis_volume_bbl" not in recon.columns:
        recon["production_basis_volume_bbl"] = recon["bopm"].fillna(0) + recon["bwpm"].fillna(0)

    if "actual_ppm" not in recon.columns:
        recon["actual_ppm"] = pd.NA

    ppm_fill_mask = (
        recon["actual_ppm"].isna()
        & recon["actual_gpd"].notna()
        & recon["production_basis_volume_bbl"].notna()
        & (recon["production_basis_volume_bbl"] > 0)
    )
    if ppm_fill_mask.any():
        recon.loc[ppm_fill_mask, "actual_ppm"] = (
            recon.loc[ppm_fill_mask, "actual_gpd"] * 1_000_000
        ) / (recon.loc[ppm_fill_mask, "production_basis_volume_bbl"] * 42)

    if "missing_target_flag" not in recon.columns:
        recon["missing_target_flag"] = recon["target_gpd"].isna().astype(int)
    else:
        recon["missing_target_flag"] = _safe_numeric(recon["missing_target_flag"]).fillna(0).astype(int)

    if "missing_actual_flag" not in recon.columns:
        recon["missing_actual_flag"] = recon["actual_gpd"].isna().astype(int)
    else:
        recon["missing_actual_flag"] = _safe_numeric(recon["missing_actual_flag"]).fillna(0).astype(int)

    if "no_production_flag" not in recon.columns:
        recon["no_production_flag"] = (recon["production_basis_volume_bbl"].fillna(0) <= 0).astype(int)
    else:
        recon["no_production_flag"] = _safe_numeric(recon["no_production_flag"]).fillna(0).astype(int)

    if "no_chemical_flag" not in recon.columns:
        recon["no_chemical_flag"] = (recon["actual_gpd"].fillna(0) <= 0).astype(int)
    else:
        recon["no_chemical_flag"] = _safe_numeric(recon["no_chemical_flag"]).fillna(0).astype(int)

    if "stale_target_flag" not in recon.columns:
        recon["stale_target_flag"] = 0
    else:
        recon["stale_target_flag"] = _safe_numeric(recon["stale_target_flag"]).fillna(0).astype(int)

    if "actual_exists_flag" not in recon.columns:
        recon["actual_exists_flag"] = recon["actual_gpd"].notna()
    if "spend_exists_flag" not in recon.columns:
        recon["spend_exists_flag"] = recon["actual_cost"].notna()
    if "target_exists_flag" not in recon.columns:
        recon["target_exists_flag"] = recon["target_gpd"].notna()

    recon["var_actual_vs_expected_gpd"] = recon.apply(
        lambda r: _calc_abs_variance(r.get("actual_gpd"), r.get("expected_gpd")),
        axis=1,
    )
    recon["var_actual_vs_target_gpd"] = recon.apply(
        lambda r: _calc_abs_variance(r.get("actual_gpd"), r.get("target_gpd")),
        axis=1,
    )
    recon["var_actual_vs_expected_pct"] = recon.apply(
        lambda r: _calc_pct_variance(r.get("actual_gpd"), r.get("expected_gpd")),
        axis=1,
    )
    recon["var_actual_vs_target_pct"] = recon.apply(
        lambda r: _calc_pct_variance(r.get("actual_gpd"), r.get("target_gpd")),
        axis=1,
    )
    recon["var_actual_ppm_vs_target_ppm"] = recon.apply(
        lambda r: _calc_pct_variance(r.get("actual_ppm"), r.get("target_ppm")),
        axis=1,
    )

    tol_df = load_chem_tolerance(settings)

    recon["feed_status"] = recon.apply(
        lambda r: _classify_feed_status(r, tol_df=tol_df),
        axis=1,
    )

    recon["operational_status"] = recon.apply(
        lambda r: classify_operational_status(
            target_rate=r.get("target_gpd"),
            actual_rate=r.get("actual_gpd"),
            chem_type=r.get("chem_type"),
            tol_df=tol_df,
            expected_rate=r.get("expected_gpd"),
            missing_target_flag=r.get("missing_target_flag"),
            missing_actual_flag=r.get("missing_actual_flag"),
            no_production_flag=r.get("no_production_flag"),
            no_chemical_flag=r.get("no_chemical_flag"),
            stale_target_flag=r.get("stale_target_flag"),
        ),
        axis=1,
    )

    recon["financial_status"] = recon.apply(
        lambda r: classify_financial_status(
            target_exists=bool(r.get("target_exists_flag")),
            spend_exists=bool(r.get("spend_exists_flag")),
            actual_exists=bool(r.get("actual_exists_flag")),
            missing_target_flag=r.get("missing_target_flag"),
            missing_actual_flag=r.get("missing_actual_flag"),
        ),
        axis=1,
    )

    recon["exception_code"] = recon.apply(assign_exception_code, axis=1)
    recon["chem_status"] = recon["feed_status"]

    if "confidence_status" not in recon.columns:
        recon["confidence_status"] = pd.NA

    keep_cols = [
        "well_id",
        "date",
        "chemical_key",
        "chem_name",
        "chem_type",
        "expected_flag",
        "target_exists_flag",
        "actual_exists_flag",
        "spend_exists_flag",
        "missing_target_flag",
        "missing_actual_flag",
        "no_production_flag",
        "no_chemical_flag",
        "stale_target_flag",
        "target_gpd",
        "expected_gpd",
        "actual_gpd",
        "target_ppm",
        "expected_ppm",
        "actual_ppm",
        "target_basis",
        "target_effective_date",
        "target_last_update_date",
        "target_age_days",
        "production_basis_volume_bbl",
        "bopm",
        "bwpm",
        "actual_cost",
        "var_actual_vs_expected_gpd",
        "var_actual_vs_target_gpd",
        "var_actual_vs_expected_pct",
        "var_actual_vs_target_pct",
        "var_actual_ppm_vs_target_ppm",
        "feed_status",
        "chem_status",
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