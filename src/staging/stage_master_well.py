from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import MASTER_WELL_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.duplicate_checks import check_duplicates


def stage_master_well(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["master_well"]))
    if not files:
        logger.warning("No master_well file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, path.name, batch.batch_id, "master_well")

    exceptions = []
    exceptions += validate_required_columns(df, MASTER_WELL_SCHEMA, "raw_master_well")
    if not exceptions:
        exceptions += check_duplicates(df, ["well_id"], "raw_master_well")

    if exceptions:
        logger.warning("Master well staging exceptions: %s", len(exceptions))

    write_table(df, staged_dir, "stg_master_well", settings)
    batch.set_row_count("stg_master_well", len(df))
    logger.info("Staged master well | rows=%s", len(df))
    return df