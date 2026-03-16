from __future__ import annotations

import pandas as pd


def summarize_esp_daily(scada_df: pd.DataFrame) -> pd.DataFrame:
    if scada_df.empty:
        return pd.DataFrame(
            columns=[
                "well_id",
                "date",
                "runtime_hr",
                "shutdown_count",
                "restart_count",
                "avg_amps",
                "amps_stddev",
                "trip_count",
                "avg_intake_pressure",
                "avg_discharge_pressure",
                "avg_frequency_hz",
                "max_motor_temp",
                "fault_family",
                "fault_count",
            ]
        )

    work = scada_df.copy()
    work["date"] = pd.to_datetime(work["timestamp"], errors="coerce").dt.date

    def _metric(metric_name: str) -> pd.DataFrame:
        return work[work["metric_name"] == metric_name].copy()

    runtime = (
        _metric("runtime_hr")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .sum()
        .rename(columns={"tag_value_num": "runtime_hr"})
    )
    shutdowns = (
        _metric("shutdown_count")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .sum()
        .rename(columns={"tag_value_num": "shutdown_count"})
    )
    restarts = (
        _metric("restart_count")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .sum()
        .rename(columns={"tag_value_num": "restart_count"})
    )
    amps_avg = (
        _metric("amps")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_amps"})
    )
    amps_sd = (
        _metric("amps")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .std()
        .rename(columns={"tag_value_num": "amps_stddev"})
    )
    trips = (
        _metric("trip_count")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .sum()
        .rename(columns={"tag_value_num": "trip_count"})
    )
    ip = (
        _metric("intake_pressure")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_intake_pressure"})
    )
    dp = (
        _metric("discharge_pressure")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_discharge_pressure"})
    )
    freq = (
        _metric("frequency_hz")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_frequency_hz"})
    )
    temp = (
        _metric("motor_temp")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .max()
        .rename(columns={"tag_value_num": "max_motor_temp"})
    )

    fault = _metric("fault_family")
    if not fault.empty:
        fault["tag_value_text"] = fault["tag_value_text"].fillna("UNKNOWN")
        fault_count = (
            fault.groupby(["well_id", "date"], as_index=False)
            .size()
            .rename(columns={"size": "fault_count"})
        )
        fault_dom = (
            fault.groupby(["well_id", "date", "tag_value_text"], as_index=False)
            .size()
            .sort_values(["well_id", "date", "size"], ascending=[True, True, False])
            .drop_duplicates(["well_id", "date"])
            .rename(columns={"tag_value_text": "fault_family"})
            [["well_id", "date", "fault_family"]]
        )
    else:
        fault_count = pd.DataFrame(columns=["well_id", "date", "fault_count"])
        fault_dom = pd.DataFrame(columns=["well_id", "date", "fault_family"])

    out = runtime.copy()
    for df in [shutdowns, restarts, amps_avg, amps_sd, trips, ip, dp, freq, temp, fault_count, fault_dom]:
        out = out.merge(df, how="outer", on=["well_id", "date"])

    return out
