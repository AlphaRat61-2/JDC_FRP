from __future__ import annotations

import pandas as pd

from src.chemistry.well_chem_program import (
    expand_well_chem_program_daily,
    load_well_chem_program,
)
from src.common.paths import FACT_PRODUCTION, get_path
from src.io.exception_store import append_exceptions
from src.io.writers import write_table


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_prod(prod: pd.DataFrame) -> pd.DataFrame:
    prod = prod.copy()

    if "date" in prod.columns:
        prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.date

    rename_map = {}
    if "oil_bbl" in prod.columns and "bopm" not in prod.columns:
        rename_map["oil_bbl"] = "bopm"
    if "water_bbl" in prod.columns and "bwpm" not in prod.columns:
        rename_map["water_bbl"] = "bwpm"
    if rename_map:
        prod = prod.rename(columns=rename_map)

    required_prod = {"well_id", "date", "bopm", "bwpm"}
    missing_prod = required_prod - set(prod.columns)
    if missing_prod:
        raise ValueError(f"FACT_PRODUCTION missing required columns: {sorted(missing_prod)}")

    prod["well_id"] = prod["well_id"].astype(str).str.strip()
    prod["bopm"] = _safe_numeric(prod["bopm"])
    prod["bwpm"] = _safe_numeric(prod["bwpm"])

    prod = prod.dropna(subset=["date"])
    prod = prod.drop_duplicates(subset=["well_id", "date"], keep="last")
    prod["production_basis_volume_bbl"] = prod["bopm"].fillna(0) + prod["bwpm"].fillna(0)

    return prod[["well_id", "date", "bopm", "bwpm", "production_basis_volume_bbl"]].copy()


def _calculate_expected_gpd_from_ppm(target_ppm, total_liquid_bbl):
    target_ppm = pd.to_numeric(target_ppm, errors="coerce")
    total_liquid_bbl = pd.to_numeric(total_liquid_bbl, errors="coerce")

    if pd.isna(target_ppm) or pd.isna(total_liquid_bbl) or total_liquid_bbl <= 0:
        return pd.NA

    return (target_ppm * total_liquid_bbl * 42) / 1_000_000


def _calculate_expected_ppm_from_gpd(expected_gpd, total_liquid_bbl):
    expected_gpd = pd.to_numeric(expected_gpd, errors="coerce")
    total_liquid_bbl = pd.to_numeric(total_liquid_bbl, errors="coerce")

    if pd.isna(expected_gpd) or pd.isna(total_liquid_bbl) or total_liquid_bbl <= 0:
        return pd.NA

    return (expected_gpd * 1_000_000) / (total_liquid_bbl * 42)


def build_expected_chem_daily(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")
    target_path = modeled_dir / "fact_chem_target_daily.csv"

    prod = pd.read_csv(FACT_PRODUCTION, dtype={"well_id": str})
    prod = _normalize_prod(prod)

    # ------------------------------------------------------------
    # Preferred source: daily target table
    # ------------------------------------------------------------
    if target_path.exists():
        target = pd.read_csv(target_path, dtype={"well_id": str, "chemical_key": str})

        if not target.empty:
            if "date" in target.columns:
                target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.date

            target["well_id"] = target["well_id"].astype(str).str.strip()
            target["chemical_key"] = target["chemical_key"].astype(str).str.strip()

            # Backward compatibility if older builds still have target_rate
            if "target_gpd" not in target.columns and "target_rate" in target.columns:
                target["target_gpd"] = pd.to_numeric(target["target_rate"], errors="coerce")
            else:
                target["target_gpd"] = _safe_numeric(target.get("target_gpd"))

            if "target_ppm" in target.columns:
                target["target_ppm"] = _safe_numeric(target["target_ppm"])
            else:
                target["target_ppm"] = pd.NA

            target = target.merge(
                prod,
                on=["well_id", "date"],
                how="left",
            )

            target["missing_production_flag"] = (
                target["bopm"].isna() & target["bwpm"].isna()
            ).astype(int)

            target["no_production_flag"] = (
                target["production_basis_volume_bbl"].fillna(0) <= 0
            ).astype(int)

            target["missing_target_flag"] = target["target_gpd"].isna().astype(int)

            # expected_gpd logic:
            # 1. Use target_gpd if present
            # 2. Else compute from target_ppm and actual daily production
            target["expected_gpd"] = target["target_gpd"]

            ppm_mask = target["expected_gpd"].isna() & target["target_ppm"].notna()
            if ppm_mask.any():
                target.loc[ppm_mask, "expected_gpd"] = target.loc[ppm_mask].apply(
                    lambda row: _calculate_expected_gpd_from_ppm(
                        row.get("target_ppm"),
                        row.get("production_basis_volume_bbl"),
                    ),
                    axis=1,
                )

            # expected_ppm logic:
            # 1. Prefer target_ppm if already supplied
            # 2. Else derive from expected_gpd and actual production denominator
            target["expected_ppm"] = target["target_ppm"]

            ppm_fill_mask = target["expected_ppm"].isna() & target["expected_gpd"].notna()
            if ppm_fill_mask.any():
                target.loc[ppm_fill_mask, "expected_ppm"] = target.loc[ppm_fill_mask].apply(
                    lambda row: _calculate_expected_ppm_from_gpd(
                        row.get("expected_gpd"),
                        row.get("production_basis_volume_bbl"),
                    ),
                    axis=1,
                )

            target["expected_calc_valid_flag"] = (
                target["expected_gpd"].notna() &
                (target["missing_target_flag"] == 0)
            ).astype(int)

            if "stale_target_flag" not in target.columns:
                target["stale_target_flag"] = 0

            expected = pd.DataFrame(
                {
                    "well_id": target["well_id"],
                    "date": target["date"],
                    "chemical_key": target["chemical_key"],
                    "expected_gpd": _safe_numeric(target["expected_gpd"]),
                    "expected_ppm": _safe_numeric(target["expected_ppm"]),
                    "expected_unit": "GPD",
                    "expected_basis": target.get("target_basis", "RATE_GPD"),
                    "expected_source": "fact_chem_target_daily",
                    "expected_source_priority": 1,
                    "expected_confidence": "HIGH",
                    "expected_status": "ACTIVE",
                    "production_basis_volume_bbl": _safe_numeric(target["production_basis_volume_bbl"]),
                    "bopm": _safe_numeric(target["bopm"]),
                    "bwpm": _safe_numeric(target["bwpm"]),
                    "missing_production_flag": target["missing_production_flag"].astype(int),
                    "missing_target_flag": target["missing_target_flag"].astype(int),
                    "no_production_flag": target["no_production_flag"].astype(int),
                    "stale_target_flag": _safe_numeric(target["stale_target_flag"]).fillna(0).astype(int),
                    "expected_calc_valid_flag": target["expected_calc_valid_flag"].astype(int),
                }
            ).drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")

            write_table(expected, modeled_dir, "fact_expected_chem_daily", settings)
            batch.set_row_count("fact_expected_chem_daily", len(expected))
            logger.info(
                "Built fact_expected_chem_daily from fact_chem_target_daily | rows=%s",
                len(expected),
            )
            return expected

    # ------------------------------------------------------------
    # Fallback source: interval program config
    # ------------------------------------------------------------
    program = load_well_chem_program()

    if program.empty:
        logger.warning("Expected chemistry cannot be built — no daily target or chem program found.")
        return pd.DataFrame()

    daily, exceptions = expand_well_chem_program_daily(program, batch_id=batch.batch_id)

    if exceptions:
        append_exceptions(exceptions)

    if daily.empty:
        logger.warning("Expected chemistry cannot be built — chem program expansion returned no rows.")
        return pd.DataFrame()

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date
    daily["well_id"] = daily["well_id"].astype(str).str.strip()
    daily["chemical_key"] = daily["chemical_key"].astype(str).str.strip()

    daily = daily.merge(
        prod,
        on=["well_id", "date"],
        how="left",
    )

    daily["missing_production_flag"] = (
        daily["bopm"].isna() & daily["bwpm"].isna()
    ).astype(int)

    daily["no_production_flag"] = (
        daily["production_basis_volume_bbl"].fillna(0) <= 0
    ).astype(int)

    expected_gpd = daily.get("default_target_gpd")
    if expected_gpd is None:
        expected_gpd = pd.Series(pd.NA, index=daily.index)
    expected_gpd = pd.to_numeric(expected_gpd, errors="coerce")

    expected = pd.DataFrame(
        {
            "well_id": daily["well_id"],
            "date": daily["date"],
            "chemical_key": daily["chemical_key"],
            "expected_gpd": expected_gpd,
            "expected_ppm": pd.NA,
            "expected_unit": "GPD",
            "expected_basis": daily.get("default_target_basis", "PROGRAM"),
            "expected_source": "well_chem_program",
            "expected_source_priority": daily.get("target_source_preference", 3),
            "expected_confidence": "MEDIUM",
            "expected_status": "ACTIVE",
            "production_basis_volume_bbl": _safe_numeric(daily["production_basis_volume_bbl"]),
            "bopm": _safe_numeric(daily["bopm"]),
            "bwpm": _safe_numeric(daily["bwpm"]),
            "missing_production_flag": daily["missing_production_flag"].astype(int),
            "missing_target_flag": expected_gpd.isna().astype(int),
            "no_production_flag": daily["no_production_flag"].astype(int),
            "stale_target_flag": 0,
            "expected_calc_valid_flag": expected_gpd.notna().astype(int),
        }
    ).drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")

    # Derive ppm where possible
    ppm_mask = expected["expected_gpd"].notna() & expected["production_basis_volume_bbl"].notna()
    if ppm_mask.any():
        expected.loc[ppm_mask, "expected_ppm"] = expected.loc[ppm_mask].apply(
            lambda row: _calculate_expected_ppm_from_gpd(
                row.get("expected_gpd"),
                row.get("production_basis_volume_bbl"),
            ),
            axis=1,
        )

    write_table(expected, modeled_dir, "fact_expected_chem_daily", settings)
    batch.set_row_count("fact_expected_chem_daily", len(expected))
    logger.info(
        "Built fact_expected_chem_daily from well_chem_program | rows=%s",
        len(expected),
    )
    return expected