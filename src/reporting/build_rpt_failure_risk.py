from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def build_rpt_failure_risk(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")
    ml_dir = project_root() / "outputs" / "ml"

    path = ml_dir / "ml_failure_risk_scored.csv"
    if not path.exists():
        logger.warning("ml_failure_risk_scored.csv not found.")
        return pd.DataFrame()

    rpt = pd.read_csv(path, dtype={"well_id": str})
    write_table(rpt, reports_dir, "rpt_failure_risk_scored", settings)
    batch.set_row_count("rpt_failure_risk_scored", len(rpt))
    logger.info("Built rpt_failure_risk_scored | rows=%s", len(rpt))
    return rpt
