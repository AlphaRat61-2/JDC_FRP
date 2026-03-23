from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _status_rank(value: str) -> int:
    if pd.isna(value):
        return 99

    value = str(value).strip().upper()

    rank_map = {
        "MISSING_TARGET": 1,
        "MISSING_ACTUAL": 2,
        "NO_PRODUCTION": 3,
        "NO_CHEMICAL": 4,
        "STALE_TARGET": 5,
        "UNDERFEED": 6,
        "OVERFEED": 7,
        "UNKNOWN": 8,
        "ON_TARGET": 9,
    }
    return rank_map.get(value, 99)


def _priority_score(row) -> int:
    score = 0

    if int(row.get("missing_target_flag", 0)) == 1:
        score += 100
    if int(row.get("missing_actual_flag", 0)) == 1:
        score += 90
    if int(row.get("no_production_flag", 0)) == 1:
        score += 70
    if int(row.get("no_chemical_flag", 0)) == 1:
        score += 80
    if int(row.get("stale_target_flag", 0)) == 1:
        score += 60

    feed_status = str(row.get("feed_status", "")).upper()
    if feed_status == "UNDERFEED":
        score += 50
    elif feed_status == "OVERFEED":
        score += 40
    elif feed_status == "UNKNOWN":
        score += 20

    var_pct = pd.to_numeric(row.get("var_actual_vs_expected_pct"), errors="coerce")
    if pd.notna(var_pct):
        score += min(int(abs(var_pct) * 100), 25)

    return score


def _build_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["well_id", "chemical_key", "date"]).reset_index(drop=True)

    out["actual_gpd_num"] = _safe_numeric(out.get("actual_gpd"))
    out["expected_gpd_num"] = _safe_numeric(out.get("expected_gpd"))

    group_keys = ["well_id", "chemical_key"]

    out["rolling_7d_actual_gpd"] = (
        out.groupby(group_keys)["actual_gpd_num"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    out["rolling_7d_expected_gpd"] = (
        out.groupby(group_keys)["expected_gpd_num"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    out["rolling_30d_actual_gpd"] = (
        out.groupby(group_keys)["actual_gpd_num"]
        .transform(lambda s: s.rolling(30, min_periods=1).sum())
    )
    out["rolling_30d_expected_gpd"] = (
        out.groupby(group_keys)["expected_gpd_num"]
        .transform(lambda s: s.rolling(30, min_periods=1).sum())
    )

    out["rolling_7d_actual_vs_expected_pct"] = pd.NA
    mask7 = out["rolling_7d_expected_gpd"].notna() & (out["rolling_7d_expected_gpd"] != 0)
    out.loc[mask7, "rolling_7d_actual_vs_expected_pct"] = (
        (out.loc[mask7, "rolling_7d_actual_gpd"] - out.loc[mask7, "rolling_7d_expected_gpd"])
        / out.loc[mask7, "rolling_7d_expected_gpd"]
    )

    out["rolling_30d_actual_vs_expected_pct"] = pd.NA
    mask30 = out["rolling_30d_expected_gpd"].notna() & (out["rolling_30d_expected_gpd"] != 0)
    out.loc[mask30, "rolling_30d_actual_vs_expected_pct"] = (
        (out.loc[mask30, "rolling_30d_actual_gpd"] - out.loc[mask30, "rolling_30d_expected_gpd"])
        / out.loc[mask30, "rolling_30d_expected_gpd"]
    )

    out["date"] = out["date"].dt.date
    out = out.drop(columns=["actual_gpd_num", "expected_gpd_num"], errors="ignore")

    return out


def build_rpt_chemistry(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    recon_path = modeled_dir / "fact_chem_recon_daily.csv"
    well_path = modeled_dir / "dim_well.csv"
    chem_path = modeled_dir / "dim_chemical.csv"

    if not recon_path.exists():
        logger.warning("fact_chem_recon_daily.csv not found.")
        return pd.DataFrame()

    recon = pd.read_csv(recon_path, dtype={"well_id": str, "chemical_key": str})

    if recon.empty:
        write_table(recon, reports_dir, "rpt_chemistry_daily", settings)
        batch.set_row_count("rpt_chemistry_daily", 0)
        logger.info("Built rpt_chemistry_daily | rows=0")
        return recon

    if "date" in recon.columns:
        recon["date"] = pd.to_datetime(recon["date"], errors="coerce").dt.date

    # ------------------------------------------------------------
    # Merge well attributes
    # ------------------------------------------------------------
    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})

        well_cols = [
            c for c in [
                "well_id",
                "well_name",
                "asset",
                "route",
                "field",
                "area",
                "well_status",
                "lift_type",
            ]
            if c in well.columns
        ]

        if well_cols:
            well = well[well_cols].drop_duplicates(subset=["well_id"], keep="last")
            recon = recon.merge(
                well,
                how="left",
                on="well_id",
            )

    # ------------------------------------------------------------
    # Merge chemical attributes
    # ------------------------------------------------------------
    if chem_path.exists():
        chem = pd.read_csv(chem_path, dtype={"chemical_key": str})

        chem_cols = [
            c for c in [
                "chemical_key",
                "normalized_chemical_name",
                "chem_type",
                "dose_basis",
                "chemical_category",
                "vendor",
            ]
            if c in chem.columns
        ]

        if chem_cols:
            chem = chem[chem_cols].drop_duplicates(subset=["chemical_key"], keep="last")
            recon = recon.merge(
                chem,
                how="left",
                on="chemical_key",
                suffixes=("", "_dim"),
            )

            if "chem_type_dim" in recon.columns:
                if "chem_type" in recon.columns:
                    recon["chem_type"] = recon["chem_type"].where(
                        recon["chem_type"].notna(),
                        recon["chem_type_dim"],
                    )
                else:
                    recon["chem_type"] = recon["chem_type_dim"]
                recon = recon.drop(columns=["chem_type_dim"])

            if "dose_basis_dim" in recon.columns:
                if "target_basis" in recon.columns:
                    recon["target_basis"] = recon["target_basis"].where(
                        recon["target_basis"].notna(),
                        recon["dose_basis_dim"],
                    )
                recon = recon.drop(columns=["dose_basis_dim"])

    # ------------------------------------------------------------
    # Numeric cleanup
    # ------------------------------------------------------------
    numeric_cols = [
        "target_gpd",
        "expected_gpd",
        "actual_gpd",
        "target_ppm",
        "expected_ppm",
        "actual_ppm",
        "actual_cost",
        "var_actual_vs_expected_gpd",
        "var_actual_vs_target_gpd",
        "var_actual_vs_expected_pct",
        "var_actual_vs_target_pct",
        "var_actual_ppm_vs_target_ppm",
        "production_basis_volume_bbl",
        "bopm",
        "bwpm",
        "target_age_days",
    ]
    for col in numeric_cols:
        if col in recon.columns:
            recon[col] = _safe_numeric(recon[col])

    flag_cols = [
        "expected_flag",
        "target_exists_flag",
        "actual_exists_flag",
        "spend_exists_flag",
        "missing_target_flag",
        "missing_actual_flag",
        "no_production_flag",
        "no_chemical_flag",
        "stale_target_flag",
    ]
    for col in flag_cols:
        if col in recon.columns:
            recon[col] = _safe_numeric(recon[col]).fillna(0).astype(int)
        else:
            recon[col] = 0

    # ------------------------------------------------------------
    # Derived report fields
    # ------------------------------------------------------------
    if "chem_status" not in recon.columns:
        if "feed_status" in recon.columns:
            recon["chem_status"] = recon["feed_status"]
        else:
            recon["chem_status"] = "UNKNOWN"

    recon["status_rank"] = recon["chem_status"].apply(_status_rank)

    recon["flag_count"] = (
        recon["missing_target_flag"]
        + recon["missing_actual_flag"]
        + recon["no_production_flag"]
        + recon["no_chemical_flag"]
        + recon["stale_target_flag"]
    )

    recon["priority_score"] = recon.apply(_priority_score, axis=1)

    if "target_last_update_date" in recon.columns:
        recon["target_last_update_date"] = pd.to_datetime(
            recon["target_last_update_date"], errors="coerce"
        ).dt.date

    if "target_effective_date" in recon.columns:
        recon["target_effective_date"] = pd.to_datetime(
            recon["target_effective_date"], errors="coerce"
        ).dt.date

    if "days_since_target_update" not in recon.columns:
        if "target_age_days" in recon.columns:
            recon["days_since_target_update"] = recon["target_age_days"]
        else:
            recon["days_since_target_update"] = pd.NA

    recon["days_missing_actual"] = pd.NA
    if "missing_actual_flag" in recon.columns:
        recon = recon.sort_values(["well_id", "chemical_key", "date"]).reset_index(drop=True)

        def _calc_missing_streak(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            streak = 0
            values = []
            for flag in group["missing_actual_flag"].fillna(0).astype(int):
                if flag == 1:
                    streak += 1
                    values.append(streak)
                else:
                    streak = 0
                    values.append(0)
            group["days_missing_actual"] = values
            return group

        recon = (
            recon.groupby(["well_id", "chemical_key"], group_keys=False)
            .apply(_calc_missing_streak)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------
    # Rolling engineering trends
    # ------------------------------------------------------------
    recon = _build_rolling_metrics(recon)

    # ------------------------------------------------------------
    # Final presentation order
    # ------------------------------------------------------------
    preferred_cols = [
        "date",
        "well_id",
        "well_name",
        "asset",
        "route",
        "field",
        "area",
        "well_status",
        "lift_type",
        "chemical_key",
        "normalized_chemical_name",
        "chem_type",
        "chemical_category",
        "target_basis",
        "chem_status",
        "feed_status",
        "operational_status",
        "financial_status",
        "exception_code",
        "status_rank",
        "priority_score",
        "flag_count",
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
        "var_actual_vs_expected_gpd",
        "var_actual_vs_target_gpd",
        "var_actual_vs_expected_pct",
        "var_actual_vs_target_pct",
        "var_actual_ppm_vs_target_ppm",
        "rolling_7d_actual_gpd",
        "rolling_7d_expected_gpd",
        "rolling_7d_actual_vs_expected_pct",
        "rolling_30d_actual_gpd",
        "rolling_30d_expected_gpd",
        "rolling_30d_actual_vs_expected_pct",
        "production_basis_volume_bbl",
        "bopm",
        "bwpm",
        "actual_cost",
        "target_effective_date",
        "target_last_update_date",
        "target_age_days",
        "days_since_target_update",
        "days_missing_actual",
        "expected_flag",
        "target_exists_flag",
        "actual_exists_flag",
        "spend_exists_flag",
        "confidence_status",
    ]
    preferred_cols = [c for c in preferred_cols if c in recon.columns]

    remaining_cols = [c for c in recon.columns if c not in preferred_cols]
    recon = recon[preferred_cols + remaining_cols].copy()

    recon = recon.drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    recon = recon.sort_values(
        ["date", "priority_score", "well_id", "chemical_key"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    write_table(recon, reports_dir, "rpt_chemistry_daily", settings)
    batch.set_row_count("rpt_chemistry_daily", len(recon))
    logger.info("Built rpt_chemistry_daily | rows=%s", len(recon))
    return recon