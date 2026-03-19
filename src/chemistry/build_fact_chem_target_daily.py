from __future__ import annotations

import pandas as pd

from src.common.paths import CHEM_RATES_RAW, FACT_PRODUCTION, MODELED_DIR
from src.common.paths import get_path


OUTPUT_FILE = MODELED_DIR / "fact_chem_target_daily.csv"


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


def build_fact_chem_target_daily() -> pd.DataFrame:
    chem = pd.read_csv(CHEM_RATES_RAW, dtype={"well_id": str})
    prod = pd.read_csv(FACT_PRODUCTION, dtype={"well_id": str})

    chem["date"] = pd.to_datetime(chem["date"], errors="coerce").dt.date
    prod["date"] = pd.to_datetime(prod["date"], errors="coerce").dt.date

    # normalize production column names if needed
    rename_map = {}
    if "oil_bbl" in prod.columns and "bopm" not in prod.columns:
        rename_map["oil_bbl"] = "bopm"
    if "water_bbl" in prod.columns and "bwpm" not in prod.columns:
        rename_map["water_bbl"] = "bwpm"
    if rename_map:
        prod = prod.rename(columns=rename_map)

    required_chem = {"well_id", "date", "target_gpd"}
    missing_chem = required_chem - set(chem.columns)
    if missing_chem:
        raise ValueError(f"CHEM_RATES_RAW missing required columns: {sorted(missing_chem)}")

    required_prod = {"well_id", "date", "bopm", "bwpm"}
    missing_prod = required_prod - set(prod.columns)
    if missing_prod:
        raise ValueError(f"FACT_PRODUCTION missing required columns: {sorted(missing_prod)}")

    df = chem.merge(
        prod[["well_id", "date", "bopm", "bwpm"]],
        on=["well_id", "date"],
        how="left",
    )

    df["target_gpd"] = pd.to_numeric(df["target_gpd"], errors="coerce")
    df["bopm"] = pd.to_numeric(df["bopm"], errors="coerce")
    df["bwpm"] = pd.to_numeric(df["bwpm"], errors="coerce")

    df["total_liquid_bbl"] = df["bopm"].fillna(0) + df["bwpm"].fillna(0)
    df["target_ppm"] = df.apply(calculate_target_ppm, axis=1)

    if "chemical_key" not in df.columns:
        df["chemical_key"] = pd.NA

    missing_key = df["chemical_key"].isna().sum()
    if missing_key > 0:
        print(f"Dropping target rows with missing chemical_key: {missing_key}")

    df = df[df["chemical_key"].notna()].copy()

    df = df[
        [
            "date",
            "well_id",
            "chemical_key",
            "chem_name",
            "chem_type",
            "target_gpd",
            "target_ppm",
            "bopm",
            "bwpm",
            "total_liquid_bbl",
        ]
    ]

    df = df.drop_duplicates(subset=["well_id", "date", "chemical_key"], keep="last")
    df = df.sort_values(["well_id", "date", "chemical_key"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"fact_chem_target_daily created: {len(df)} rows")
    return df


if __name__ == "__main__":
    build_fact_chem_target_daily()