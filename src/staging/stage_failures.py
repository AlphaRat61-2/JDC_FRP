from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import FAILURE_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.field_checks import parse_date_column


def stage_failures(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["failures"]))
    if not files:
        logger.warning("No failures file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, path.name, batch.batch_id, "failures")

    exceptions = validate_required_columns(df, FAILURE_SCHEMA, "raw_failure_date")
    if exceptions:
        logger.warning("Failure schema exceptions: %s", len(exceptions))

    df = parse_date_column(df, "install_date")
    df = parse_date_column(df, "fail_date")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["failure_cost"] = pd.to_numeric(df["failure_cost"], errors="coerce")

    write_table(df, staged_dir, "stg_failures", settings)
    batch.set_row_count("stg_failures", len(df))
    logger.info("Staged failures | rows=%s", len(df))
    return df