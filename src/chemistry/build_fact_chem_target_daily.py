import pandas as pd
from pathlib import Path

from src.common.paths import CHEM_RATES_RAW, FACT_PRODUCTION, MODELED_DIR


OUTPUT_FILE = MODELED_DIR / "fact_chem_target_daily.csv"


def calculate_target_ppm(row):

    oil = row.get("bopm", 0)
    water = row.get("bwpm", 0)

    total_liquid = oil + water

    if total_liquid == 0:
        return 0

    gpd = row.get("target_gpd", 0)

    ppm = (gpd * 1_000_000) / (total_liquid * 42)

    return ppm


def build_fact_chem_target_daily():

    chem = pd.read_csv(CHEM_RATES_RAW)

    prod = pd.read_csv(FACT_PRODUCTION)

    df = chem.merge(
        prod,
        on=["well_id", "date"],
        how="left"
    )

    df["target_ppm"] = df.apply(calculate_target_ppm, axis=1)

    df["total_liquid_bbl"] = df["bopm"].fillna(0) + df["bwpm"].fillna(0)

    df = df[
        [
            "date",
            "well_id",
            "chem_name",
            "chem_type",
            "target_gpd",
            "target_ppm",
            "bopm",
            "bwpm",
            "total_liquid_bbl"
        ]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"fact_chem_target_daily created: {len(df)} rows")


if __name__ == "__main__":

    build_fact_chem_target_daily()