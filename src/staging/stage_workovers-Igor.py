from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import get_path
from src.common.exceptions import make_exception
from src.io.exception_store import append_exceptions
from src.io.readers import add_source_metadata, read_csv_file
from src.io.writers import write_table


REQUIRED_WORKOVER_COLS = {
    "well_id",
    "event_date",
    "work_type",
    "repair_detail",
    "parts_replaced",
    "vendor",
    "cost",
    "comment",
}


def stage_workovers(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["workovers"]))
    if not files:
        logger.warning("No workover file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, file_name=path.name, batch_id=batch.batch_id, source_name="workovers")

    exceptions: list[dict] = []

    missing = REQUIRED_WORKOVER_COLS - set(df.columns)
    for col in sorted(missing):
        exceptions.append(
            make_exception(
                table_name="raw_workovers",
                record_key="",
                exception_category="SCHEMA",
                exception_code="MISSING_REQUIRED_COLUMN",
                severity="CRITICAL",
                message=f"Missing required workover column: {col}",
                batch_id=batch.batch_id,
            )
        )

    if missing:
        append_exceptions(exceptions)
        return pd.DataFrame()

    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    bad_dates = df[df["event_date"].isna()]
    for idx, row in bad_dates.iterrows():
        exceptions.append(
            make_exception(
                table_name="raw_workovers",
                record_key=f"{row.get('well_id', '')}|{idx}",
                exception_category="FIELD",
                exception_code="INVALID_DATE",
                severity="WARNING",
                message="Could not parse workover event_date.",
                batch_id=batch.batch_id,
            )
        )

    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    if exceptions:
        append_exceptions(exceptions)

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date

    write_table(df, staged_dir, "stg_workovers", settings)
    batch.set_row_count("stg_workovers", len(df))
    logger.info("Staged workovers | rows=%s", len(df))
    return df
