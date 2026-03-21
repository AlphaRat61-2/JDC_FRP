from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].astype(str).str.strip()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    return df


def _log_duplicate_sample(df: pd.DataFrame, logger, name: str) -> None:
    dupes = df.duplicated(subset=["well_id", "date"], keep=False)
    count = int(dupes.sum())

    if count == 0:
        return

    sample = (
        df.loc[dupes]
        .sort_values(["well_id", "date"])
        .head(20)
        .copy()
    )

    logger.warning(
        "%s duplicate well_id/date rows found: %s | sample:\n%s",
        name,
        count,
        sample.to_string(index=False),
    )


def build_fact_production_daily(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    path = staged_dir / "stg_production_daily.csv"
    if not path.exists():
        logger.warning("stg_production_daily.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"], dtype={"well_id": str})
    df = _standardize_keys(df)

    required = ["well_id", "date", "oil_bbl", "gas_mcf", "water_bbl", "source_system"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("stg_production_daily missing required columns: %s", missing)
        return pd.DataFrame()

    bad_dates = int(df["date"].isna().sum())
    if bad_dates > 0:
        logger.warning(
            "stg_production_daily contains rows with invalid date values: %s. "
            "Dropping rows with null date.",
            bad_dates,
        )
        df = df.loc[df["date"].notna()].copy()

    for col in ["oil_bbl", "gas_mcf", "water_bbl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicate rows first.
    exact_dupe_count = int(df.duplicated(keep="first").sum())
    if exact_dupe_count > 0:
        logger.warning(
            "stg_production_daily exact duplicate rows found: %s. "
            "Dropping exact duplicates before fact build.",
            exact_dupe_count,
        )
        df = df.drop_duplicates().copy()

    _log_duplicate_sample(df, logger, "stg_production_daily")

    dupes = df.duplicated(subset=["well_id", "date"], keep=False)
    if dupes.any():
        # Resolve remaining duplicate keys by keeping the most complete row.
        # This handles cases like one valid measured row plus one blank row.
        work = df.copy()

        non_null_count = (
            work[["oil_bbl", "gas_mcf", "water_bbl"]]
            .notna()
            .sum(axis=1)
        )
        total_volume = (
            work["oil_bbl"].fillna(0)
            + work["gas_mcf"].fillna(0)
            + work["water_bbl"].fillna(0)
        )

        work["_nonnull_count"] = non_null_count
        work["_total_volume"] = total_volume

        work = (
            work.sort_values(
                ["well_id", "date", "_nonnull_count", "_total_volume"],
                ascending=[True, True, False, False],
            )
            .drop_duplicates(subset=["well_id", "date"], keep="first")
            .drop(columns=["_nonnull_count", "_total_volume"])
            .copy()
        )

        remaining = work.duplicated(subset=["well_id", "date"], keep=False)
        if remaining.any():
            sample = (
                work.loc[remaining]
                .sort_values(["well_id", "date"])
                .head(20)
            )
            raise ValueError(
                "Unable to resolve duplicate well_id/date rows in fact_production_daily.\n"
                + sample.to_string(index=False)
            )

        logger.warning(
            "stg_production_daily duplicate well_id/date rows were resolved by keeping the most complete row per well/date."
        )

        df = work

    gas_factor = settings["units"]["boe_gas_mcf_per_boe"]

    fact = df[["well_id", "date", "oil_bbl", "gas_mcf", "water_bbl", "source_system"]].copy()
    fact["total_fluid_bbl"] = fact["oil_bbl"].fillna(0) + fact["water_bbl"].fillna(0)
    fact["boe"] = fact["oil_bbl"].fillna(0) + (fact["gas_mcf"].fillna(0) / gas_factor)
    fact["prod_data_status"] = "MEASURED"
    fact["prod_source"] = fact["source_system"]
    fact = fact.drop(columns=["source_system"])

    remaining_fact_dupes = fact.duplicated(subset=["well_id", "date"], keep=False)
    if remaining_fact_dupes.any():
        sample = (
            fact.loc[remaining_fact_dupes]
            .sort_values(["well_id", "date"])
            .head(20)
        )
        raise ValueError(
            "fact_production_daily still contains duplicate well_id/date rows before write.\n"
            + sample.to_string(index=False)
        )

    fact["date"] = pd.to_datetime(fact["date"], errors="coerce").dt.date

    write_table(fact, modeled_dir, "fact_production_daily", settings)
    batch.set_row_count("fact_production_daily", len(fact))
    logger.info("Built fact_production_daily | rows=%s", len(fact))
    return fact