from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import CHEMICAL_RATES_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.field_checks import parse_date_column, validate_numeric_nonnegative


def stage_chemical_rates(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["chemical_rates"]))
    if not files:
        logger.warning("No chemical_rates file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, path.name, batch.batch_id, "chemical_rates")

    exceptions = validate_required_columns(df, CHEMICAL_RATES_SCHEMA, "raw_chemical_rates")
    if exceptions:
        logger.warning("Chemical rates schema exceptions: %s", len(exceptions))

    df = parse_date_column(df, "date")
    df["target_gpd"] = pd.to_numeric(df["target_gpd"], errors="coerce")
    exceptions += validate_numeric_nonnegative(df, "target_gpd", "raw_chemical_rates")

    write_table(df, staged_dir, "stg_chemical_rates", settings)
    batch.set_row_count("stg_chemical_rates", len(df))
    logger.info("Staged chemical rates | rows=%s", len(df))
    return df