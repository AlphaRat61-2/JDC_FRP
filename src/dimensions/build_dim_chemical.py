from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_dim_chemical(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")
    config_dir = get_path(settings, "incoming").parents[0] / "config"

    frames = []

    for fname in ["stg_chemical_rates.csv", "stg_chemical_cost.csv"]:
        path = staged_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df[["chem_name", "chem_type"]].copy())

    if not frames:
        logger.warning("No chemistry staging files found for dim_chemical.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True).drop_duplicates()

    chem_map_path = config_dir / "chem_name_map.csv"
    if chem_map_path.exists():
        chem_map = pd.read_csv(chem_map_path)
        dim = raw.merge(chem_map, how="left", on="chem_name")
    else:
        dim = raw.copy()
        dim["chemical_key"] = dim["chem_name"].str.upper().str.replace(" ", "_", regex=False)
        dim["normalized_chemical_name"] = dim["chem_name"]
        dim["dose_basis"] = None

    keep_cols = [
        c for c in [
            "chemical_key",
            "normalized_chemical_name",
            "chem_type",
            "dose_basis"
        ] if c in dim.columns
    ]
    dim = dim[keep_cols].drop_duplicates()

    write_table(dim, modeled_dir, "dim_chemical", settings)
    batch.set_row_count("dim_chemical", len(dim))
    logger.info("Built dim_chemical | rows=%s", len(dim))
    return dim