from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import CHEMICAL_COST_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.field_checks import parse_date_column


def stage_chemical_cost(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["chemical_cost"]))
    if not files:
        logger.warning("No chemical_cost file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, path.name, batch.batch_id, "chemical_cost")

    exceptions = validate_required_columns(df, CHEMICAL_COST_SCHEMA, "raw_chemical_cost")
    if exceptions:
        logger.warning("Chemical cost schema exceptions: %s", len(exceptions))

    df = parse_date_column(df, "date")
    for col in ["qty", "unit_cost", "actual_cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    write_table(df, staged_dir, "stg_chemical_cost", settings)
    batch.set_row_count("stg_chemical_cost", len(df))
    logger.info("Staged chemical cost | rows=%s", len(df))
    return df