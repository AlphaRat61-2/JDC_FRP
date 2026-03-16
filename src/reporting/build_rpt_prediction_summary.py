from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def build_rpt_prediction_summary(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")
    ml_dir = project_root() / "outputs" / "ml"

    risk_path = ml_dir / "ml_failure_risk_scored.csv"
    perf_path = ml_dir / "ml_model_performance.csv"

    rows: list[dict] = []

    if risk_path.exists():
        risk = pd.read_csv(risk_path, dtype={"well_id": str})
        rows.extend(
            [
                {"metric": "scored_rows", "value": len(risk)},
                {
                    "metric": "critical_risk_wells",
                    "value": int((risk["risk_bucket"] == "CRITICAL").sum())
                    if "risk_bucket" in risk.columns
                    else 0,
                },
                {
                    "metric": "high_risk_wells",
                    "value": int((risk["risk_bucket"] == "HIGH").sum())
                    if "risk_bucket" in risk.columns
                    else 0,
                },
                {
                    "metric": "avg_failure_risk_30d",
                    "value": float(risk["failure_risk_30d"].mean())
                    if "failure_risk_30d" in risk.columns
                    else None,
                },
            ]
        )

    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        for _, row in perf.iterrows():
            rows.append(
                {
                    "metric": f"model_{row['metric']}",
                    "value": row["value"],
                }
            )

    rpt = pd.DataFrame(rows)
    write_table(rpt, reports_dir, "rpt_prediction_summary", settings)
    batch.set_row_count("rpt_prediction_summary", len(rpt))
    logger.info("Built rpt_prediction_summary | rows=%s", len(rpt))
    return rpt
