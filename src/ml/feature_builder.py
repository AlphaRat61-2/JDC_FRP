from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def _ml_output_dir() -> str:
    return "ml"


def _safe_group_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    return (
        df.sort_values([group_col, order_col])
        .groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )


def _safe_group_rolling_sum(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    return (
        df.sort_values([group_col, order_col])
        .groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).sum())
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
    prod["date"] = pd.to_datetime(prod["date"], errors="coerce")

    feat = prod[
        ["well_id", "date", "oil_bbl", "gas_mcf", "water_bbl", "boe", "total_fluid_bbl"]
    ].copy()

    denom = (feat["oil_bbl"].fillna(0) + feat["water_bbl"].fillna(0)).replace(0, pd.NA)
    feat["water_cut"] = feat["water_bbl"].fillna(0) / denom

    feat["production_change_7d"] = (
        feat.sort_values(["well_id", "date"])
        .groupby("well_id")["boe"]
        .transform(lambda s: s - s.shift(7))
    )
    feat["boe_7d_avg"] = _safe_group_rolling_mean(feat, "well_id", "date", "boe", 7)

    if chem_path.exists():
        chem = pd.read_csv(chem_path, parse_dates=["date"], dtype={"well_id": str})
        chem["date"] = pd.to_datetime(chem["date"], errors="coerce")
        chem["chem_exception_flag"] = chem["exception_code"].fillna("UNKNOWN") != "FULL_MATCH"
        chem["spend_amt"] = pd.to_numeric(chem.get("spend"), errors="coerce").fillna(0)

        chem_daily = chem.groupby(["well_id", "date"], as_index=False).agg(
            chem_exception_days=("chem_exception_flag", "sum"),
            spend_daily=("spend_amt", "sum"),
        )

        feat = feat.merge(chem_daily, how="left", on=["well_id", "date"])
        feat["chem_exception_days"] = feat["chem_exception_days"].fillna(0)
        feat["spend_daily"] = feat["spend_daily"].fillna(0)

        feat["chem_exception_7d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "chem_exception_days", 7
        )
        feat["spend_30d"] = _safe_group_rolling_sum(
            feat, "well_id", "date", "spend_daily", 30
        )
    else:
        feat["chem_exception_7d"] = 0
        feat["spend_30d"] = 0

    if scada_path.exists():
        scada = pd.read_csv(scada_path, parse_dates=["date"], dtype={"well_id": str})
        scada["date"] = pd.to_datetime(scada["date"], errors="coerce")

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
        feat = feat.merge(scada_daily, how="left", on=["well_id", "date"])

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
    else:
        feat["runtime_7d_avg"] = None
        feat["fillage_7d_avg"] = None
        feat["fillage_decline_7d"] = None
        feat["trip_count_7d"] = 0
        feat["shutdown_count_7d"] = 0
        feat["deterioration_score"] = None
        feat["pre_failure_flag"] = False

    if fail_path.exists():
        fail = pd.read_csv(fail_path, parse_dates=["fail_date"], dtype={"well_id": str})
        fail["fail_date"] = pd.to_datetime(fail["fail_date"], errors="coerce")
        fail = fail.sort_values(["well_id", "fail_date"])

        feat = feat.sort_values(["well_id", "date"])

        last_fail = feat.merge(
            fail[["well_id", "fail_date"]],
            how="left",
            on="well_id",
        )
        last_fail = last_fail[last_fail["fail_date"] <= last_fail["date"]].copy()
        last_fail = last_fail.groupby(["well_id", "date"], as_index=False)["fail_date"].max()
        feat = feat.merge(last_fail, how="left", on=["well_id", "date"])
        feat["days_since_last_failure"] = (feat["date"] - feat["fail_date"]).dt.days

        fail_daily = (
            fail.groupby(["well_id", "fail_date"], as_index=False)
            .size()
            .rename(columns={"size": "failure_events", "fail_date": "date"})
        )
        feat = feat.merge(fail_daily, how="left", on=["well_id", "date"])
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
        wo["event_date"] = pd.to_datetime(wo["event_date"], errors="coerce")

        wo_daily = wo.groupby(["well_id", "event_date"], as_index=False).agg(
            workover_events=("workover_event_id", "count"),
            workover_cost_daily=("cost", "sum"),
        ).rename(columns={"event_date": "date"})

        feat = feat.merge(wo_daily, how="left", on=["well_id", "date"])
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
        feat = feat.merge(
            well[["well_id", "asset", "route", "lift_type", "equipment_profile_id"]],
            how="left",
            on="well_id",
        )

    feat["date"] = pd.to_datetime(feat["date"], errors="coerce").dt.date

    write_table(feat, ml_dir, "ml_feature_well_daily", settings)
    batch.set_row_count("ml_feature_well_daily", len(feat))
    logger.info("Built ml_feature_well_daily | rows=%s", len(feat))
    return feat

