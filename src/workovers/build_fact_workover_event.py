from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_fact_workover_event(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    path = staged_dir / "stg_workovers.csv"
    if not path.exists():
        logger.warning("stg_workovers.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"well_id": str})
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    fact = df.copy()
    fact["workover_event_id"] = (
        fact["well_id"].astype(str)
        + "_"
        + fact["event_date"].dt.strftime("%Y%m%d").fillna("NA")
        + "_"
        + fact.index.astype(str)
    )

    keep_cols = [
        "workover_event_id",
        "well_id",
        "event_date",
        "work_type",
        "repair_detail",
        "parts_replaced",
        "vendor",
        "cost",
        "comment",
        "source_system",
        "source_file_name",
        "batch_id",
    ]
    keep_cols = [c for c in keep_cols if c in fact.columns]
    fact = fact[keep_cols].copy()
    fact["event_date"] = pd.to_datetime(fact["event_date"], errors="coerce").dt.date

    write_table(fact, modeled_dir, "fact_workover_event", settings)
    batch.set_row_count("fact_workover_event", len(fact))
    logger.info("Built fact_workover_event | rows=%s", len(fact))
    return fact
