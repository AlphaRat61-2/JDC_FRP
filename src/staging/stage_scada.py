from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import get_path
from src.io.exception_store import append_exceptions
from src.io.readers import add_source_metadata, read_csv_file
from src.io.writers import write_table
from src.common.exceptions import make_exception


def stage_scada(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob("scada*.csv"))
    if not files:
        logger.warning("No scada file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, file_name=path.name, batch_id=batch.batch_id, source_name="scada")

    required_cols = {"well_id", "timestamp", "tag_name", "tag_value"}
    missing = required_cols - set(df.columns)
    exceptions: list[dict] = []

    if missing:
        for col in sorted(missing):
            exceptions.append(
                make_exception(
                    table_name="raw_scada",
                    record_key="",
                    exception_category="SCHEMA",
                    exception_code="MISSING_REQUIRED_COLUMN",
                    severity="CRITICAL",
                    message=f"Missing required SCADA column: {col}",
                    batch_id=batch.batch_id,
                )
            )
        append_exceptions(settings, exceptions, logger)
        return pd.DataFrame()

    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    bad_ts = df[df["timestamp"].isna()]
    for idx, row in bad_ts.iterrows():
        exceptions.append(
            make_exception(
                table_name="raw_scada",
                record_key=f"{row.get('well_id', '')}|{idx}",
                exception_category="FIELD",
                exception_code="INVALID_TIMESTAMP",
                severity="WARNING",
                message="Could not parse SCADA timestamp.",
                batch_id=batch.batch_id,
            )
        )

    df["tag_value_num"] = pd.to_numeric(df["tag_value"], errors="coerce")
    df["tag_value_text"] = df["tag_value"].astype(str)

    if exceptions:
        append_exceptions(settings, exceptions, logger)

    write_table(df, staged_dir, "stg_scada", settings)
    batch.set_row_count("stg_scada", len(df))
    logger.info("Staged scada | rows=%s", len(df))
    return df
