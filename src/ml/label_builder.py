from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def build_ml_label_failure(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    ml_dir = project_root() / "outputs" / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    prod_path = modeled_dir / "fact_production_daily.csv"
    fail_path = modeled_dir / "fact_failure_event.csv"

    if not prod_path.exists():
        logger.warning("fact_production_daily.csv not found for label creation.")
        return pd.DataFrame()

    base = pd.read_csv(prod_path, parse_dates=["date"], dtype={"well_id": str})[
        ["well_id", "date"]
    ].drop_duplicates()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")

    if not fail_path.exists():
        labels = base.copy()
        labels["failure_within_7d"] = False
        labels["failure_within_14d"] = False
        labels["failure_within_30d"] = False
        labels["failure_type_future"] = None
        labels["date"] = labels["date"].dt.date
        write_table(labels, ml_dir, "ml_label_failure", settings)
        batch.set_row_count("ml_label_failure", len(labels))
        logger.info("Built ml_label_failure | rows=%s", len(labels))
        return labels

    fail = pd.read_csv(fail_path, parse_dates=["fail_date"], dtype={"well_id": str})
    fail["fail_date"] = pd.to_datetime(fail["fail_date"], errors="coerce")

    def _future_label(row_date: pd.Timestamp, fail_dates: pd.Series, days: int) -> bool:
        return ((fail_dates > row_date) & (fail_dates <= row_date + pd.Timedelta(days=days))).any()

    out_rows = []
    for well_id, g in base.groupby("well_id"):
        fail_well = fail[fail["well_id"] == well_id].copy()
        fail_dates = fail_well["fail_date"]

        for _, row in g.iterrows():
            d = row["date"]
            future = fail_well[fail_well["fail_date"] > d].sort_values("fail_date")
            future_type = future.iloc[0]["failure_type"] if not future.empty else None

            out_rows.append(
                {
                    "well_id": well_id,
                    "date": d.date(),
                    "failure_within_7d": _future_label(d, fail_dates, 7),
                    "failure_within_14d": _future_label(d, fail_dates, 14),
                    "failure_within_30d": _future_label(d, fail_dates, 30),
                    "failure_type_future": future_type,
                }
            )

    labels = pd.DataFrame(out_rows)
    write_table(labels, ml_dir, "ml_label_failure", settings)
    batch.set_row_count("ml_label_failure", len(labels))
    logger.info("Built ml_label_failure | rows=%s", len(labels))
    return labels
