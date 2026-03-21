from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.paths import get_path
from src.io.readers import read_csv_file, add_source_metadata
from src.io.writers import write_table
from src.validation.schemas import CHEMICAL_COST_SCHEMA
from src.validation.schema_checks import validate_required_columns
from src.validation.field_checks import parse_date_column


def _ensure_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype("string").str.strip()
    return df


def _normalize_line_category(value) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {"CHEMICAL", "CHEM"}:
        return "CHEMICAL"
    if text in {"EQUIPMENT", "EQUIP", "RENTAL", "SERVICE"}:
        return "EQUIPMENT"
    if text in {"DISCOUNT", "CREDIT", "REBATE"}:
        return "DISCOUNT"

    return text or None


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

    df = _ensure_string_columns(
        df,
        [
            "asset",
            "route",
            "well_id",
            "well_name",
            "vendor",
            "line_category",
            "chem_name",
            "chem_type",
            "equipment",
            "truck_treat",
            "uom",
        ],
    )

    df["well_id"] = df["well_id"].astype("string").str.strip()

    df["line_category"] = df["line_category"].apply(_normalize_line_category).astype("string")

    for col in ["qty", "unit_cost", "actual_cost"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Helpful defaults for downstream logic
    if "vendor" not in df.columns:
        df["vendor"] = pd.NA
    if "truck_treat" not in df.columns:
        df["truck_treat"] = pd.NA
    if "uom" not in df.columns:
        df["uom"] = pd.NA

    # Log category counts so we can verify the new structure is being used
    if "line_category" in df.columns:
        category_counts = (
            df["line_category"]
            .fillna("UNSPECIFIED")
            .value_counts(dropna=False)
            .to_dict()
        )
        logger.info("Chemical cost line_category counts | %s", category_counts)

    write_table(df, staged_dir, "stg_chemical_cost", settings)
    batch.set_row_count("stg_chemical_cost", len(df))
    logger.info("Staged chemical cost | rows=%s", len(df))
    return df