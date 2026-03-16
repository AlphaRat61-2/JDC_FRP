from __future__ import annotations

import pandas as pd


def summarize_rod_pump_daily(scada_df: pd.DataFrame) -> pd.DataFrame:
    if scada_df.empty:
        return pd.DataFrame(
            columns=[
                "well_id",
                "date",
                "runtime_hr",
                "shutdown_count",
                "restart_count",
                "avg_fillage_pct",
                "min_fillage_pct",
                "low_fillage_event_count",
                "avg_spm",
                "avg_polished_rod_load",
                "dyno_card_count",
                "dominant_dyno_condition",
                "dyno_severity_score",
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
    fillage_avg = (
        _metric("fillage_pct")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_fillage_pct"})
    )
    fillage_min = (
        _metric("fillage_pct")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .min()
        .rename(columns={"tag_value_num": "min_fillage_pct"})
    )
    fillage_low = (
        _metric("fillage_pct")
        .assign(low_flag=lambda d: d["tag_value_num"] < 55)
        .groupby(["well_id", "date"], as_index=False)["low_flag"]
        .sum()
        .rename(columns={"low_flag": "low_fillage_event_count"})
    )
    spm = (
        _metric("spm")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_spm"})
    )
    prl = (
        _metric("polished_rod_load")
        .groupby(["well_id", "date"], as_index=False)["tag_value_num"]
        .mean()
        .rename(columns={"tag_value_num": "avg_polished_rod_load"})
    )

    dyno = _metric("dyno_condition")
    if not dyno.empty:
        dyno["tag_value_text"] = dyno["tag_value_text"].fillna("UNKNOWN")
        dyno_cards = (
            dyno.groupby(["well_id", "date"], as_index=False)
            .size()
            .rename(columns={"size": "dyno_card_count"})
        )
        dyno_dom = (
            dyno.groupby(["well_id", "date", "tag_value_text"], as_index=False)
            .size()
            .sort_values(["well_id", "date", "size"], ascending=[True, True, False])
            .drop_duplicates(["well_id", "date"])
            .rename(columns={"tag_value_text": "dominant_dyno_condition"})
            [["well_id", "date", "dominant_dyno_condition"]]
        )
    else:
        dyno_cards = pd.DataFrame(columns=["well_id", "date", "dyno_card_count"])
        dyno_dom = pd.DataFrame(columns=["well_id", "date", "dominant_dyno_condition"])

    out = runtime.copy()
    for df in [shutdowns, restarts, fillage_avg, fillage_min, fillage_low, spm, prl, dyno_cards, dyno_dom]:
        out = out.merge(df, how="outer", on=["well_id", "date"])

    out["dyno_severity_score"] = 0.0
    out.loc[out["avg_fillage_pct"].fillna(100) < 55, "dyno_severity_score"] += 1.0
    out.loc[out["shutdown_count"].fillna(0) > 3, "dyno_severity_score"] += 1.0

    return out
