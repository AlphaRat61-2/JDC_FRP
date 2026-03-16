from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def append_batch_history(settings: dict, batch, logger) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    path = modeled_dir / "pipeline_batch_history.csv"

    row = {
        "batch_id": batch.batch_id,
        "run_type": batch.run_type,
        "run_start_ts": batch.start_ts.isoformat() if batch.start_ts else None,
        "run_end_ts": batch.end_ts.isoformat() if batch.end_ts else None,
        "status": batch.status,
        "notes": " | ".join(batch.notes) if batch.notes else None,
        "row_counts_json": str(batch.row_counts) if batch.row_counts else None,
    }
    new_df = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    write_table(combined, modeled_dir, "pipeline_batch_history", settings)
    logger.info("Updated pipeline_batch_history | rows=%s", len(combined))
    return combined
