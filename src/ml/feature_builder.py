from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def _ml_output_dir() -> str:
    return "ml"


def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    return df


def _safe_group_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    ordered = df.sort_values([group_col, order_col]).copy()
    return (
        ordered.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        .reindex(ordered.index)
        .sort_index()
    )


def _safe_group_rolling_sum(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    ordered = df.sort_values([group_col, order_col]).copy()
    return (
        ordered.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).sum())
        .reindex(ordered.index)
        .sort_index()
    )


def _log_duplicate_sample(
    df: pd.DataFrame,
    logger,
    name: str,
    keys: list[str],
) -> None:
    dupes = df.duplicated(subset=keys, keep=False)
    count = int(dupes.sum())

    if count == 0:
        return

    sample = (
        df.loc[dupes]
        .sort_values(keys)
        .head(20)
        .copy()
    )
    logger.warning(
        "%s duplicate %s rows found: %s | sample:\n%s",
        name,
        "/".join(keys),
        count,
        sample.to_string(index=False),
    )


def _assert_unique_keys(
    df: pd.DataFrame,
    logger,
    name: str,
    keys: list[str],
) -> pd.DataFrame:
    _log_duplicate_sample(df, logger, name, keys)
    dupes = df.duplicated(subset=keys, keep=False)
    if dupes.any():
        raise ValueError(
            f"{name} still contains duplicate key rows on {keys} after cleanup."
        )
    return df


def _coerce_bool_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype("boolean")
        .fillna(False)
        .astype(int)
    )


def build_ml_feature_well_daily(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    ml_dir = project_root() / "outputs" / _ml_output_dir()
    ml_dir.mkdir(parents=True, exist_ok=True)

    prod_path = modeled_dir / "fact_production_daily.csv"
    chem_path = modeled_dir / "fact_chem_recon_daily.csv"
    scada_path = modeled_dir / "fact_scada_daily.csv"
    fail_path = modeled_dir / "fact_failure_event.csv"
    workover_path = modeled_dir / "fact_workover_event.csv"
    well_path = modeled_dir / "dim_well.csv"

    if not prod_path.exists():
        logger.warning("fact_production_daily.csv not found.")
        return pd.DataFrame()

    prod = pd.read_csv(prod_path, parse_dates=["date"], dtype={"well_id": str})
    prod = _standardize_keys(prod)

    prod_required = ["well_id", "date"]
    missing_prod = [c for c in prod_required if c not in prod.columns]
    if missing_prod:
        logger.error("fact_production_daily missing required columns: %s", missing_prod)
        return pd.DataFrame()

    bad_prod_dates = int(prod["date"].isna().sum())
    if bad_prod_dates > 0:
        logger.warning(
            "fact_production_daily contains rows with invalid date values: %s. "
            "Dropping rows with null date.",
            bad_prod_dates,
        )
        prod = prod.loc[prod["date"].notna()].copy()

    for col in ["oil_bbl", "gas_mcf", "water_bbl", "boe", "total_fluid_bbl"]:
        if col in prod.columns:
            prod[col] = pd.to_numeric(prod[col], errors="coerce")

    if "gfflap" in prod.columns:
        prod["gfflap"] = pd.to_numeric(prod["gfflap"], errors="coerce")
    else:
        prod["gfflap"] = float("nan")

    _log_duplicate_sample(prod, logger, "fact_production_daily", ["well_id", "date"])

    prod_dupes = prod.duplicated(subset=["well_id", "date"], keep=False)
    if prod_dupes.any():
        agg_spec = {}
        for col in ["oil_bbl", "gas_mcf", "water_bbl", "boe", "total_fluid_bbl"]:
            if col in prod.columns:
                agg_spec[col] = "sum"
        if "gfflap" in prod.columns:
            agg_spec["gfflap"] = "max"

        prod = (
            prod.groupby(["well_id", "date"], as_index=False)
            .agg(agg_spec)
            .copy()
        )
        logger.warning(
            "fact_production_daily duplicates were aggregated to one row per well_id/date with gfflap preserved."
        )

    prod = _assert_unique_keys(prod, logger, "fact_production_daily", ["well_id", "date"])

    base_cols = [
        "well_id",
        "date",
        "oil_bbl",
        "gas_mcf",
        "water_bbl",
        "boe",
        "total_fluid_bbl",
        "gfflap",
    ]
    base_cols = [c for c in base_cols if c in prod.columns]

    feat = prod[base_cols].copy()
    feat["well_id"] = feat["well_id"].astype(str).str.strip()

    denom = (feat["oil_bbl"].fillna(0) + feat["water_bbl"].fillna(0)).replace(0, pd.NA)
    feat["water_cut"] = feat["water_bbl"].fillna(0) / denom

    feat["production_change_7d"] = (
        feat.sort_values(["well_id", "date"])
        .groupby("well_id")["boe"]
        .transform(lambda s: s - s.shift(7))
    )
    feat["boe_7d_avg"] = _safe_group_rolling_mean(feat, "well_id", "date", "boe", 7)

    if "gfflap" in feat.columns:

        feat = feat.sort_values(["well_id", "date"])

        # Last measured value
        feat["gfflap_last"] = (
            feat.groupby("well_id")["gfflap"].ffill()
        )

        # Last test date
        feat["gfflap_last_date"] = feat["date"].where(feat["gfflap"].notna())

        feat["gfflap_last_date"] = (
            feat.groupby("well_id")["gfflap_last_date"].ffill()
        )

        # Days since last test
        feat["gfflap_days_since"] = (
            (feat["date"] - feat["gfflap_last_date"]).dt.days
        )

        # Previous test value
        feat["gfflap_prev"] = (
            feat.groupby("well_id")["gfflap"].shift(1)
        )

        # Change from last test
        feat["gfflap_change"] = feat["gfflap"] - feat["gfflap_prev"]

        # Cleanup temp columns
        feat = feat.drop(columns=["gfflap_last_date", "gfflap_prev"])

    else:
        feat["gfflap_last"] = pd.NA
        feat["gfflap_days_since"] = pd.NA
        feat["gfflap_change"] = pd.NA


    if chem_path.exists():
        chem = pd.read_csv(chem_path, parse_dates=["date"], dtype={"well_id": str})
        chem = _standardize_keys(chem)

        bad_chem_dates = int(chem["date"].isna().sum()) if "date" in chem.columns else 0
        if bad_chem_dates > 0:
            logger.warning(
                "fact_chem_recon_daily contains rows with invalid date values: %s. "
                "Dropping rows with null date.",
                bad_chem_dates,
            )
            chem = chem.loc[chem["date"].notna()].copy()

        exception_code = (
            chem["exception_code"]
            if "exception_code" in chem.columns
            else pd.Series(index=chem.index, dtype="object")
        )
        target_exists = (
            chem["target_exists_flag"]
            if "target_exists_flag" in chem.columns
            else pd.Series(False, index=chem.index, dtype="boolean")
        )
        actual_exists = (
            chem["actual_exists_flag"]
            if "actual_exists_flag" in chem.columns
            else pd.Series(False, index=chem.index, dtype="boolean")
        )
        operational_status = (
            chem["operational_status"]
            if "operational_status" in chem.columns
            else pd.Series("", index=chem.index, dtype="object")
        )

        chem["chem_exception_flag"] = exception_code.fillna("UNKNOWN") != "FULL_MATCH"
        chem["spend_amt"] = (
            pd.to_numeric(chem["spend"], errors="coerce").fillna(0)
            if "spend" in chem.columns
            else 0.0
        )

        chem["chem_target_missing_flag"] = _coerce_bool_flag(~target_exists.astype("boolean"))
        chem["chem_actual_missing_flag"] = _coerce_bool_flag(~actual_exists.astype("boolean"))
        chem["chem_noncompliant_flag"] = operational_status.fillna("").isin(
            ["UNDER_TREATED", "OVER_TREATED", "MISSING_TARGET", "MISSING_ACTUAL"]
        ).astype(int)

        chem["chem_miss_flag"] = (
            (chem["chem_target_missing_flag"] > 0) | (chem["chem_actual_missing_flag"] > 0)
        ).astype(int)

        chem_daily = chem.groupby(["well_id", "date"], as_index=False).agg(
            chem_exception_days=("chem_exception_flag", "sum"),
            chem_noncompliant_days=("chem_noncompliant_flag", "sum"),
            chem_miss_days=("chem_miss_flag", "sum"),
            chem_target_missing_days=("chem_target_missing_flag", "sum"),
            chem_actual_missing_days=("chem_actual_missing_flag", "sum"),
            spend_daily=("spend_amt", "sum"),
        )

        chem_daily = _assert_unique_keys(
            chem_daily, logger, "fact_chem_recon_daily_agg", ["well_id", "date"]
        )

        feat = feat.merge(
            chem_daily,
            how="left",
            on=["well_id", "date"],
            validate="one_to_one",
        )

        for col in [
            "chem_exception_days",
            "chem_noncompliant_days",
            "chem_miss_days",
            "chem_target_missing_days",
            "chem_actual_missing_days",
            "spend_daily",
        ]:
            feat[col] = feat[col].fillna(0)

        feat["chem_exception_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_exception_days", 7
        )
        feat["chem_noncompliant_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_noncompliant_days", 7
        )
        feat["chem_miss_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_miss_days", 7
        )
        feat["chem_target_missing_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_target_missing_days", 7
        )
        feat["chem_actual_missing_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_actual_missing_days", 7
        )
        feat["spend_30d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "spend_daily", 30
        )
    else:
        feat["chem_exception_7d"] = 0
        feat["chem_noncompliant_7d"] = 0
        feat["chem_miss_7d"] = 0
        feat["chem_target_missing_7d"] = 0
        feat["chem_actual_missing_7d"] = 0
        feat["spend_30d"] = 0

    if scada_path.exists():
        scada = pd.read_csv(scada_path, parse_dates=["date"], dtype={"well_id": str})
        scada = _standardize_keys(scada)

        bad_scada_dates = int(scada["date"].isna().sum()) if "date" in scada.columns else 0
        if bad_scada_dates > 0:
            logger.warning(
                "fact_scada_daily contains rows with invalid date values: %s. "
                "Dropping rows with null date.",
                bad_scada_dates,
            )
            scada = scada.loc[scada["date"].notna()].copy()

        if "deterioration_score" not in scada.columns:
            scada["deterioration_score"] = 0.0

        if "pre_failure_flag" not in scada.columns:
            scada["pre_failure_flag"] = 0

        agg_dict = {
            "runtime_hr": "mean",
            "shutdown_count": "sum",
            "deterioration_score": "mean",
            "pre_failure_flag": "max",
        }
        if "avg_fillage_pct" in scada.columns:
            agg_dict["avg_fillage_pct"] = "mean"
        if "trip_count" in scada.columns:
            agg_dict["trip_count"] = "sum"

        scada_daily = scada.groupby(["well_id", "date"], as_index=False).agg(agg_dict)
        scada_daily = _assert_unique_keys(
            scada_daily, logger, "fact_scada_daily_agg", ["well_id", "date"]
        )

        feat = feat.merge(
            scada_daily,
            how="left",
            on=["well_id", "date"],
            validate="one_to_one",
        )

        feat["runtime_7d_avg"] = _safe_group_rolling_mean(
            feat.fillna({"runtime_hr": 0}), "well_id", "date", "runtime_hr", 7
        )

        if "avg_fillage_pct" in feat.columns:
            feat["fillage_7d_avg"] = _safe_group_rolling_mean(
                feat.fillna({"avg_fillage_pct": 0}),
                "well_id",
                "date",
                "avg_fillage_pct",
                7,
            )
            feat["fillage_decline_7d"] = (
                feat.sort_values(["well_id", "date"])
                .groupby("well_id")["avg_fillage_pct"]
                .transform(lambda s: s - s.shift(7))
            )
        else:
            feat["fillage_7d_avg"] = None
            feat["fillage_decline_7d"] = None

        if "trip_count" in feat.columns:
            feat["trip_count_7d"] = _safe_group_rolling_sum(
                feat.fillna({"trip_count": 0}), "well_id", "date", "trip_count", 7
            )
        else:
            feat["trip_count_7d"] = 0

        feat["shutdown_count_7d"] = _safe_group_rolling_sum(
            feat.fillna({"shutdown_count": 0}),
            "well_id",
            "date",
            "shutdown_count",
            7,
        )

    if fail_path.exists():
        fail = pd.read_csv(fail_path, parse_dates=["fail_date"], dtype={"well_id": str})
        fail["well_id"] = fail["well_id"].astype(str).str.strip()
        fail["fail_date"] = pd.to_datetime(fail["fail_date"], errors="coerce").dt.normalize()
        fail = fail.loc[fail["fail_date"].notna()].copy()
        fail = fail.sort_values(["well_id", "fail_date"])

        feat = feat.sort_values(["well_id", "date"]).copy()

        last_fail = feat.merge(
            fail[["well_id", "fail_date"]],
            how="left",
            on="well_id",
        )
        last_fail = last_fail.loc[last_fail["fail_date"] <= last_fail["date"]].copy()
        last_fail = (
            last_fail.groupby(["well_id", "date"], as_index=False)["fail_date"]
            .max()
            .copy()
        )

        feat = feat.merge(
            last_fail,
            how="left",
            on=["well_id", "date"],
            validate="one_to_one",
        )
        feat["days_since_last_failure"] = (feat["date"] - feat["fail_date"]).dt.days

        fail_daily = (
            fail.groupby(["well_id", "fail_date"], as_index=False)
            .size()
            .rename(columns={"size": "failure_events", "fail_date": "date"})
        )

        fail_daily = _assert_unique_keys(
            fail_daily, logger, "fact_failure_event_daily", ["well_id", "date"]
        )

        feat = feat.merge(
            fail_daily,
            how="left",
            on=["well_id", "date"],
            validate="one_to_one",
        )
        feat["failure_events"] = feat["failure_events"].fillna(0)
        feat["failures_last_90d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "failure_events", 90
        )
        feat = feat.drop(columns=["fail_date"])
    else:
        feat["days_since_last_failure"] = None
        feat["failures_last_90d"] = 0

    if workover_path.exists():
        wo = pd.read_csv(workover_path, dtype={"well_id": str})
        wo["well_id"] = wo["well_id"].astype(str).str.strip()
        wo["event_date"] = pd.to_datetime(wo["event_date"], errors="coerce").dt.normalize()
        wo = wo.loc[wo["event_date"].notna()].copy()

        if "cost" in wo.columns:
            wo["cost"] = pd.to_numeric(wo["cost"], errors="coerce").fillna(0)
        else:
            wo["cost"] = 0.0

        workover_id_col = "workover_event_id" if "workover_event_id" in wo.columns else None

        if workover_id_col is not None:
            wo_daily = wo.groupby(["well_id", "event_date"], as_index=False).agg(
                workover_events=(workover_id_col, "count"),
                workover_cost_daily=("cost", "sum"),
            )
        else:
            wo_daily = wo.groupby(["well_id", "event_date"], as_index=False).agg(
                workover_events=("well_id", "size"),
                workover_cost_daily=("cost", "sum"),
            )

        wo_daily = wo_daily.rename(columns={"event_date": "date"})
        wo_daily = _assert_unique_keys(
            wo_daily, logger, "fact_workover_event_daily", ["well_id", "date"]
        )

        feat = feat.merge(
            wo_daily,
            how="left",
            on=["well_id", "date"],
            validate="one_to_one",
        )
        feat["workover_events"] = feat["workover_events"].fillna(0)
        feat["workover_cost_daily"] = feat["workover_cost_daily"].fillna(0)

        feat["workovers_last_90d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "workover_events", 90
        )
        feat["workover_cost_90d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "workover_cost_daily", 90
        )
    else:
        feat["workovers_last_90d"] = 0
        feat["workover_cost_90d"] = 0

    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        well["well_id"] = well["well_id"].astype(str).str.strip()

        for col, default in {
            "asset": "UNKNOWN",
            "route": "UNKNOWN",
            "lift_type": "UNKNOWN",
            "equipment_profile_id": "EQP_UNKNOWN",
        }.items():
            if col not in well.columns:
                well[col] = default

        well = well[["well_id", "asset", "route", "lift_type", "equipment_profile_id"]].copy()
        _log_duplicate_sample(well, logger, "dim_well", ["well_id"])

        if well.duplicated(subset=["well_id"], keep=False).any():
            well = (
                well.sort_values(["well_id"])
                .drop_duplicates(subset=["well_id"], keep="last")
                .copy()
            )
            logger.warning(
                "dim_well duplicates were removed by keeping the last row per well_id."
            )

        feat = feat.merge(
            well,
            how="left",
            on="well_id",
            validate="many_to_one",
        )
    else:
        feat["asset"] = "UNKNOWN"
        feat["route"] = "UNKNOWN"
        feat["lift_type"] = "UNKNOWN"
        feat["equipment_profile_id"] = "EQP_UNKNOWN"

    feat["asset"] = feat["asset"].fillna("UNKNOWN")
    feat["route"] = feat["route"].fillna("UNKNOWN")
    feat["lift_type"] = feat["lift_type"].fillna("UNKNOWN")
    feat["equipment_profile_id"] = feat["equipment_profile_id"].fillna("EQP_UNKNOWN")

    feat = _standardize_keys(feat)
    feat = _assert_unique_keys(feat, logger, "ml_feature_well_daily_final", ["well_id", "date"])

    feat["date"] = pd.to_datetime(feat["date"], errors="coerce").dt.date

    feat["lift_type_known_flag"] = (feat["lift_type"] != "UNKNOWN").astype(int)
    feat["equipment_profile_known_flag"] = (
        feat["equipment_profile_id"] != "EQP_UNKNOWN"
    ).astype(int)

    write_table(feat, ml_dir, "ml_feature_well_daily", settings)
    batch.set_row_count("ml_feature_well_daily", len(feat))
    logger.info("Built ml_feature_well_daily | rows=%s", len(feat))
    return feat