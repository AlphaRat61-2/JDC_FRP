from __future__ import annotations

import pandas as pd

from src.common.paths import project_root
from src.io.writers import write_table


def _ml_dir():
    path = project_root() / "outputs" / "ml"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_training_dataset(settings, logger, batch) -> pd.DataFrame:
    ml_dir = _ml_dir()

    feat_path = ml_dir / "ml_feature_well_daily.csv"
    label_path = ml_dir / "ml_label_failure.csv"

    if not feat_path.exists() or not label_path.exists():
        logger.warning("Missing ML feature or label table.")
        return pd.DataFrame()

    feat = pd.read_csv(feat_path, parse_dates=["date"], dtype={"well_id": str})
    lab = pd.read_csv(label_path, parse_dates=["date"], dtype={"well_id": str})

    ds = feat.merge(lab, how="inner", on=["well_id", "date"])
    ds = ds.sort_values(["date", "well_id"]).reset_index(drop=True)

    write_table(ds, ml_dir, "ml_training_dataset", settings)
    batch.set_row_count("ml_training_dataset", len(ds))
    logger.info("Built ml_training_dataset | rows=%s", len(ds))
    return ds


def split_train_test_by_time(
    df: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    work = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(work) * (1 - test_fraction))

    train_df = work.iloc[:split_idx].copy()
    test_df = work.iloc[split_idx:].copy()
    return train_df, test_df
