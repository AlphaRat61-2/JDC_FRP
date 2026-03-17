import pandas as pd
from pathlib import Path

from src.common.paths import MODELED_DIR


INPUT_FILE = MODELED_DIR / "stg_scada.csv"
OUTPUT_FILE = MODELED_DIR / "fact_scada_daily.csv"


NUMERIC_TAGS = {
    "runtime_hr",
    "shutdown_count",
    "restart_count",
    "fillage_pct",
    "spm",
    "polished_rod_load",
    "amps",
    "trip_count",
    "intake_pressure",
    "discharge_pressure",
    "frequency_hz",
    "motor_temp",
}

TEXT_TAGS = {
    "dyno_condition",
    "fault_family",
}


def _build_numeric_daily(df: pd.DataFrame) -> pd.DataFrame:
    num = df[df["tag_name"].isin(NUMERIC_TAGS)].copy()

    if num.empty:
        return pd.DataFrame()

    daily = (
        num.groupby(["well_id", "date", "tag_name"], as_index=False)["tag_value_num"]
        .mean()
    )

    wide = daily.pivot_table(
        index=["well_id", "date"],
        columns="tag_name",
        values="tag_value_num",
        aggfunc="mean"
    ).reset_index()

    wide.columns.name = None
    return wide


def _build_text_daily(df: pd.DataFrame) -> pd.DataFrame:
    txt = df[df["tag_name"].isin(TEXT_TAGS)].copy()

    if txt.empty:
        return pd.DataFrame(columns=["well_id", "date"])

    txt = txt.sort_values(["well_id", "date", "timestamp"])

    last_txt = (
        txt.groupby(["well_id", "date", "tag_name"], as_index=False)
        .last()[["well_id", "date", "tag_name", "tag_value_text"]]
    )

    wide = last_txt.pivot_table(
        index=["well_id", "date"],
        columns="tag_name",
        values="tag_value_text",
        aggfunc="first"
    ).reset_index()

    wide.columns.name = None
    return wide


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "runtime_hr" not in out.columns:
        out["runtime_hr"] = 0.0

    if "shutdown_count" not in out.columns:
        out["shutdown_count"] = 0.0

    if "restart_count" not in out.columns:
        out["restart_count"] = 0.0

    if "trip_count" not in out.columns:
        out["trip_count"] = 0.0

    if "fillage_pct" not in out.columns:
        out["fillage_pct"] = pd.NA

    out["avg_fillage_pct"] = out["fillage_pct"]

    det = pd.Series(0.0, index=out.index)

    det += out["shutdown_count"].fillna(0) * 0.8
    det += out["trip_count"].fillna(0) * 1.2
    det += ((100 - out["fillage_pct"].fillna(100)).clip(lower=0) / 25.0)

    if "motor_temp" in out.columns:
        det += ((out["motor_temp"].fillna(0) - 180).clip(lower=0) / 20.0)

    out["deterioration_score"] = det.round(3)
    out["pre_failure_flag"] = out["deterioration_score"] >= 3.0

    return out


def build_fact_scada_daily(settings=None, logger=None, batch=None):
    if not INPUT_FILE.exists():
        msg = f"Missing staged SCADA file: {INPUT_FILE}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return pd.DataFrame()

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        msg = "Staged SCADA is empty."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return pd.DataFrame()

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date

    numeric_daily = _build_numeric_daily(df)
    text_daily = _build_text_daily(df)

    if numeric_daily.empty and text_daily.empty:
        msg = "No SCADA daily rows could be built."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return pd.DataFrame()

    if numeric_daily.empty:
        fact = text_daily.copy()
    elif text_daily.empty:
        fact = numeric_daily.copy()
    else:
        fact = numeric_daily.merge(text_daily, on=["well_id", "date"], how="outer")

    fact = _derive_features(fact)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fact.to_csv(OUTPUT_FILE, index=False)

    if batch is not None:
        try:
            batch.set_row_count("fact_scada_daily", len(fact))
        except Exception:
            pass

    msg = f"Built fact_scada_daily | rows={len(fact)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return fact


if __name__ == "__main__":
    build_fact_scada_daily()