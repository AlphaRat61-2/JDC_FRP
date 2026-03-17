import pandas as pd
from pathlib import Path

from src.common.paths import RAW_DATA_DIR, MODELED_DIR


INPUT_FILE = RAW_DATA_DIR / "scada_data.csv"
OUTPUT_FILE = MODELED_DIR / "stg_scada.csv"


def stage_scada(settings=None, logger=None, batch=None):
    if not INPUT_FILE.exists():
        msg = f"SCADA file not found: {INPUT_FILE}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return pd.DataFrame()

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["well_id", "timestamp", "tag_name", "tag_value"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"SCADA file missing required columns: {missing}")

    df = df.copy()

    df["well_id"] = df["well_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["tag_name"] = df["tag_name"].astype(str).str.strip()

    df["tag_value_raw"] = df["tag_value"]
    df["tag_value_num"] = pd.to_numeric(df["tag_value"], errors="coerce")
    df["tag_value_text"] = df["tag_value"].astype(str).str.strip()
    df["date"] = df["timestamp"].dt.date

    df = df.dropna(subset=["well_id", "timestamp", "tag_name"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    if batch is not None:
        try:
            batch.set_row_count("stg_scada", len(df))
        except Exception:
            pass

    msg = f"Staged scada | rows={len(df)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return df


if __name__ == "__main__":
    stage_scada()