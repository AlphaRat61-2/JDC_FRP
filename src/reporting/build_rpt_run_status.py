from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_rpt_run_status(settings, logger, batch) -> pd.DataFrame:
    reports_dir = get_path(settings, "reports")

    row = {
        "batch_id": batch.batch_id,
        "run_type": batch.run_type,
        "status": batch.status,
        "run_start_ts": batch.start_ts.isoformat() if batch.start_ts else None,
        "run_end_ts": batch.end_ts.isoformat() if batch.end_ts else None,
        "row_counts_json": str(batch.row_counts) if batch.row_counts else None,
        "notes": " | ".join(batch.notes) if batch.notes else None,
    }
    rpt = pd.DataFrame([row])

    write_table(rpt, reports_dir, "rpt_run_status", settings)
    batch.set_row_count("rpt_run_status", len(rpt))
    logger.info("Built rpt_run_status | rows=%s", len(rpt))
    return rpt
