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


def _standardize_production_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def _ensure_production_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_defaults = {
        "oil_bbl": pd.NA,
        "gas_mcf": pd.NA,
        "water_bbl": pd.NA,
        "gfflap": pd.NA,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _log_duplicate_sample(df: pd.DataFrame, logger, table_name: str) -> None:
    dupes = df.duplicated(subset=["well_id", "date"], keep=False)
    count = int(dupes.sum())

    if count == 0:
        return

    sample_cols = [c for c in [
        "asset",
        "route",
        "well_id",
        "well_name",
        "date",
        "oil_bbl",
        "gas_mcf",
        "water_bbl",
        "gfflap",
        "source_file_name",
        "batch_id",
        "source_system",
    ] if c in df.columns]

    sample = (
        df.loc[dupes, sample_cols]
        .sort_values(["well_id", "date"])
        .head(20)
        .copy()
    )

    logger.warning(
        "%s duplicate well_id/date rows found: %s | sample:\n%s",
        table_name,
        count,
        sample.to_string(index=False),
    )


def _collapse_production_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()

    _log_duplicate_sample(df, logger, "raw_production_daily")

    dupes = df.duplicated(subset=["well_id", "date"], keep=False)
    if not dupes.any():
        return df

    # Prefer rows with real production volumes.
    df["_nonnull_prod_count"] = df[["oil_bbl", "gas_mcf", "water_bbl"]].notna().sum(axis=1)
    df["_prod_total"] = (
        df["oil_bbl"].fillna(0)
        + df["gas_mcf"].fillna(0)
        + df["water_bbl"].fillna(0)
    )
    df["_has_gfflap"] = df["gfflap"].notna().astype(int)

    def first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else pd.NA

    collapsed = (
        df.sort_values(
            ["well_id", "date", "_nonnull_prod_count", "_prod_total", "_has_gfflap"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["well_id", "date"], as_index=False)
        .agg(
            asset=("asset", first_non_null) if "asset" in df.columns else ("well_id", "first"),
            route=("route", first_non_null) if "route" in df.columns else ("well_id", "first"),
            well_name=("well_name", first_non_null) if "well_name" in df.columns else ("well_id", "first"),
            oil_bbl=("oil_bbl", first_non_null),
            gas_mcf=("gas_mcf", first_non_null),
            water_bbl=("water_bbl", first_non_null),
            gfflap=("gfflap", first_non_null),
            source_file_name=("source_file_name", first_non_null) if "source_file_name" in df.columns else ("well_id", "first"),
            batch_id=("batch_id", first_non_null) if "batch_id" in df.columns else ("well_id", "first"),
            source_system=("source_system", first_non_null) if "source_system" in df.columns else ("well_id", "first"),
        )
        .copy()
    )

    remaining = collapsed.duplicated(subset=["well_id", "date"], keep=False)
    if remaining.any():
        sample = (
            collapsed.loc[remaining]
            .sort_values(["well_id", "date"])
            .head(20)
        )
        raise ValueError(
            "Unable to collapse duplicate raw_production_daily rows to one row per well_id/date.\n"
            + sample.to_string(index=False)
        )

    logger.warning(
        "raw_production_daily duplicate well_id/date rows were collapsed to one row per well/date while preserving gfflap."
    )

    return collapsed


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
    df = _ensure_production_columns(df)
    df = parse_date_column(df, "date")
    df = _standardize_production_keys(df)

    for col in ["oil_bbl", "gas_mcf", "water_bbl", "gfflap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["oil_bbl", "gas_mcf", "water_bbl"]:
        exceptions += validate_numeric_nonnegative(df, col, "raw_production_daily")

    # Drop exact duplicate rows first.
    exact_dupe_count = int(df.duplicated(keep="first").sum())
    if exact_dupe_count > 0:
        logger.warning(
            "raw_production_daily exact duplicate rows found: %s. "
            "Dropping exact duplicates before staging.",
            exact_dupe_count,
        )
        df = df.drop_duplicates().copy()

    # Collapse remaining duplicate well/date rows while preserving gfflap.
    if df.duplicated(subset=["well_id", "date"], keep=False).any():
        df = _collapse_production_duplicates(df, logger)

    # Final duplicate check should now be informational only.
    exceptions += check_duplicates(df, ["well_id", "date"], "raw_production_daily")

    write_table(df, staged_dir, "stg_production_daily", settings)
    batch.set_row_count("stg_production_daily", len(df))
    logger.info("Staged production | rows=%s", len(df))
    return df