from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_bad_actors(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    fail_path = modeled_dir / "fact_failure_event.csv"
    chem_path = modeled_dir / "fact_chem_recon_daily.csv"
    scada_path = modeled_dir / "fact_scada_daily.csv"
    workover_path = modeled_dir / "fact_workover_event.csv"
    well_path = modeled_dir / "dim_well.csv"

    out = pd.DataFrame()

    if fail_path.exists():
        fail = pd.read_csv(fail_path, dtype={"well_id": str})
        fail_agg = fail.groupby("well_id", as_index=False).agg(
            failures=("failure_event_id", "count"),
            failure_cost=("failure_cost", "sum"),
            avg_run_life=("run_life_days", "mean"),
        )
        out = fail_agg

    if chem_path.exists():
        chem = pd.read_csv(chem_path, dtype={"well_id": str})
        chem_agg = chem.groupby("well_id", as_index=False).agg(
            chem_exception_days=("exception_code", lambda s: (pd.Series(s).fillna("UNKNOWN") != "FULL_MATCH").sum())
        )
        out = chem_agg if out.empty else out.merge(chem_agg, how="outer", on="well_id")

    if scada_path.exists():
        scada = pd.read_csv(scada_path, dtype={"well_id": str})
        scada_agg = scada.groupby("well_id", as_index=False).agg(
            surveillance_alert_days=("pre_failure_flag", lambda s: pd.Series(s).fillna(False).astype(bool).sum()),
            avg_deterioration_score=("deterioration_score", "mean"),
        )
        out = scada_agg if out.empty else out.merge(scada_agg, how="outer", on="well_id")

    if workover_path.exists():
        wo = pd.read_csv(workover_path, dtype={"well_id": str})
        wo_agg = wo.groupby("well_id", as_index=False).agg(
            workover_count=("workover_event_id", "count"),
            workover_cost=("cost", "sum"),
        )
        out = wo_agg if out.empty else out.merge(wo_agg, how="outer", on="well_id")

    if out.empty:
        logger.warning("No data available to build rpt_bad_actors.")
        return out

    out = out.fillna(0)
    out["risk_score"] = (
        out.get("failures", 0) * 3.0
        + out.get("chem_exception_days", 0) * 0.10
        + out.get("surveillance_alert_days", 0) * 1.5
        + out.get("avg_deterioration_score", 0) * 2.0
        + out.get("workover_count", 0) * 1.25
    )
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)

    if well_path.exists():
        well = pd.read_csv(well_path, dtype={"well_id": str})
        out = out.merge(
            well[["well_id", "well_name", "asset", "route", "lift_type"]],
            how="left",
            on="well_id",
        )

    write_table(out, reports_dir, "rpt_bad_actors", settings)
    batch.set_row_count("rpt_bad_actors", len(out))
    logger.info("Built rpt_bad_actors | rows=%s", len(out))
    return out

