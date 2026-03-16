from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_month_status(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    reports_dir = get_path(settings, "reports")

    path = modeled_dir / "fact_month_status.csv"
    if not path.exists():
        logger.warning("fact_month_status.csv not found.")
        return pd.DataFrame()

    rpt = pd.read_csv(path)
    write_table(rpt, reports_dir, "rpt_month_status", settings)
    batch.set_row_count("rpt_month_status", len(rpt))
    logger.info("Built rpt_month_status | rows=%s", len(rpt))
    return rpt
