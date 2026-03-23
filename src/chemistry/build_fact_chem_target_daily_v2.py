from __future__ import annotations

import pandas as pd

from src.common.paths import CHEM_RATES_RAW, FACT_PRODUCTION, MODELED_DIR


OUTPUT_FILE = MODELED_DIR / "fact_chem_target_daily.csv"

# Keep config local and simple for now.
# Can be moved to a central config later if desired.
STALE_TARGET_DAYS = 30


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def calculate_target_ppm(row) -> float:
    oil = pd.to_numeric(row.get("bopm"), errors="coerce")
    water = pd.to_numeric(row.get("bwpm"), errors="coerce")
    gpd = pd.to_numeric(row.get("target_gpd"), errors="coerce")

    oil = 0 if pd.isna(oil) else oil
    water = 0 if pd.isna(water) else water
    gpd = 0 if pd.isna(gpd) else gpd

    total_liquid = oil + water
    if total_liquid <= 0:
        return 0.0

    return (gpd * 1_000_000) / (total_liquid * 42)


def _normalize_production(prod: pd.DataFrame) -> pd.DataFrame:
    prod = prod.copy()

    prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.normalize()

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
    prod = prod.sort_values(["well_id", "date"]).reset_index(drop=True)

    return prod


def _normalize_targets(chem: pd.DataFrame) -> pd.DataFrame:
    chem = chem.copy()

    chem["date"] = pd.to_datetime(chem["date"], errors="coerce").dt.normalize()
    chem["well_id"] = chem["well_id"].astype(str).str.strip()

    required_chem = {"well_id", "date", "target_gpd"}
    missing_chem = required_chem - set(chem.columns)
    if missing_chem:
        raise ValueError(f"CHEM_RATES_RAW missing required columns: {sorted(missing_chem)}")

    if "chemical_key" not in chem.columns:
        chem["chemical_key"] = pd.NA
    if "chem_name" not in chem.columns:
        chem["chem_name"] = pd.NA
    if "chem_type" not in chem.columns:
        chem["chem_type"] = pd.NA

    chem["chemical_key"] = chem["chemical_key"].astype("string").str.strip()
    chem["chem_name"] = chem["chem_name"].astype("string").str.strip()
    chem["chem_type"] = chem["chem_type"].astype("string").str.strip()
    chem["target_gpd"] = _safe_numeric(chem["target_gpd"])

    chem = chem.dropna(subset=["date"])

    missing_key = chem["chemical_key"].isna().sum()
    if missing_key > 0:
        print(f"Dropping target rows with missing chemical_key: {missing_key}")

    chem = chem[chem["chemical_key"].notna()].copy()

    # One target change record per well / chemical / effective date
    chem = chem.drop_duplicates(subset=["well_id", "chemical_key", "date"], keep="last")

    chem["target_effective_date"] = chem["date"]
    chem["target_last_update_date"] = chem["date"]
    chem["target_basis"] = "RATE_GPD"
    chem["target_active_flag"] = 1

    chem = chem.sort_values(["well_id", "chemical_key", "date"]).reset_index(drop=True)

    return chem


def _build_daily_target_state(prod: pd.DataFrame, chem: pd.DataFrame) -> pd.DataFrame:
    """
    Expand sparse target updates into daily target state by carrying the most recent
    effective target forward across production dates for each well + chemical.
    """
    if chem.empty or prod.empty:
        return pd.DataFrame()

    # Daily calendar comes from production dates so the target fact aligns to engineering daily grain.
    prod_dates = prod[["well_id", "date", "bopm", "bwpm"]].copy()

    # Cross production dates with chemicals that exist for the well.
    well_chems = chem[["well_id", "chemical_key", "chem_name", "chem_type"]].drop_duplicates().copy()

    calendar = prod_dates.merge(well_chems, on="well_id", how="inner")
    if calendar.empty:
        return pd.DataFrame()

    calendar = calendar.sort_values(["well_id", "chemical_key", "date"]).reset_index(drop=True)

    target_cols = [
        "well_id",
        "chemical_key",
        "date",
        "target_gpd",
        "target_effective_date",
        "target_last_update_date",
        "target_basis",
        "target_active_flag",
    ]

    merged_frames = []

    for (well_id, chemical_key), cal_grp in calendar.groupby(["well_id", "chemical_key"], dropna=False):
        tgt_grp = chem[
            (chem["well_id"] == well_id) &
            (chem["chemical_key"] == chemical_key)
        ][target_cols].sort_values("date")

        if tgt_grp.empty:
            continue

        cal_grp = cal_grp.sort_values("date").copy()

        # Carry most recent target forward to each production date.
        merged = pd.merge_asof(
            cal_grp,
            tgt_grp,
            on="date",
            by=None,
            direction="backward",
            allow_exact_matches=True,
        )

        # Drop dates before first effective target.
        merged = merged[merged["target_effective_date"].notna()].copy()
        if merged.empty:
            continue

        merged_frames.append(merged)

    if not merged_frames:
        return pd.DataFrame()

    df = pd.concat(merged_frames, ignore_index=True)

    return df


def build_fact_chem_target_daily() -> pd.DataFrame:
    chem = pd.read_csv(CHEM_RATES_RAW, dtype={"well_id": str})
    prod = pd.read_csv(FACT_PRODUCTION, dtype={"well_id": str})

    chem = _normalize_targets(chem)
    prod = _normalize_production(prod)

    df = _build_daily_target_state(prod=prod, chem=chem)

    if df.empty:
        print("fact_chem_target_daily created: 0 rows")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        return df

    df["total_liquid_bbl"] = df["bopm"].fillna(0) + df["bwpm"].fillna(0)
    df["target_ppm"] = df.apply(calculate_target_ppm, axis=1)

    df["target_age_days"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["target_last_update_date"])
    ).dt.days

    df["stale_target_flag"] = (df["target_age_days"] > STALE_TARGET_DAYS).astype(int)

    df = df[
        [
            "date",
            "well_id",
            "chemical_key",
            "chem_name",
            "chem_type",
            "target_gpd",
            "target_ppm",
            "target_basis",
            "target_active_flag",
            "target_effective_date",
            "target_last_update_date",
            "target_age_days",
            "stale_target_flag",
            "bopm",
            "bwpm",
            "total_liquid_bbl",
        ]
    ].copy()

    # Convert dates back to date objects for consistency with current pipeline outputs.
    for col in ["date", "target_effective_date", "target_last_update_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    df = df.drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    df = df.sort_values(["well_id", "date", "chemical_key"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"fact_chem_target_daily created: {len(df)} rows")
    return df


if __name__ == "__main__":
    build_fact_chem_target_daily()