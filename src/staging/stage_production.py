from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import PRODUCTION_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.field_checks import parse_date_column, validate_numeric_nonnegative
from src.validation.duplicate_checks import check_duplicates




def rename_production_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "bopd": "oil_bbl",
        "mcfd": "gas_mcf",
        "bwpd": "water_bbl",
    }
    return df.rename(columns=rename_map)


def stage_production(settings, logger, batch) -> pd.DataFrame:
    incoming_dir = get_path(settings, "incoming")
    staged_dir = get_path(settings, "staged")

    files = sorted(incoming_dir.glob(settings["file_patterns"]["production"]))
    if not files:
        logger.warning("No production file found.")
        return pd.DataFrame()

    path: Path = files[-1]
    df = read_csv_file(path)
    df = add_source_metadata(df, path.name, batch.batch_id, "production")
 
    exceptions = validate_required_columns(df, PRODUCTION_SCHEMA, "raw_production_daily")
    if exceptions:
        logger.warning("Production schema exceptions: %s", len(exceptions))

    df = rename_production_columns(df)
    df = parse_date_column(df, "date")

    for col in ["oil_bbl", "gas_mcf", "water_bbl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        exceptions += validate_numeric_nonnegative(df, col, "raw_production_daily")

    exceptions += check_duplicates(df, ["well_id", "date"], "raw_production_daily")

    write_table(df, staged_dir, "stg_production_daily", settings)
    batch.set_row_count("stg_production_daily", len(df))
    logger.info("Staged production | rows=%s", len(df))
    return df