from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_dim_well(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    path = staged_dir / "stg_master_well.csv"
    if not path.exists():
        logger.warning("stg_master_well.csv not found.")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"well_id": str})
    dim = df[["well_id", "well_name", "asset", "route"]].drop_duplicates().copy()

    dim["lift_type"] = None
    dim["well_status"] = None
    dim["equipment_profile_id"] = None
    dim["is_active_flag"] = True

    write_table(dim, modeled_dir, "dim_well", settings)
    batch.set_row_count("dim_well", len(dim))
    logger.info("Built dim_well | rows=%s", len(dim))
    return dim