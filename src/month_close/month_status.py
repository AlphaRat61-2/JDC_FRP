from __future__ import annotations

import pandas as pd

from src.common.paths import get_path, project_root
from src.io.writers import write_table


def _load_month_lock() -> pd.DataFrame:
    path = project_root() / "config" / "month_lock.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"month_key": str})
        if "lock_flag" in df.columns:
            df["lock_flag"] = (
                df["lock_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            )
        return df
    return pd.DataFrame(columns=["month_key", "lock_flag", "lock_reason"])


def _month_key_from_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m")


def build_fact_month_status(settings, logger, batch) -> pd.DataFrame:
    modeled_dir = get_path(settings, "modeled")

    prod_path = modeled_dir / "fact_production_daily.csv"
    chem_path = modeled_dir / "fact_chem_recon_daily.csv"
    fail_path = modeled_dir / "fact_failure_event.csv"
    scada_path = modeled_dir / "fact_scada_daily.csv"

    month_rows: list[pd.DataFrame] = []

    if prod_path.exists():
        prod = pd.read_csv(prod_path, parse_dates=["date"])
        if not prod.empty:
            x = prod[["date"]].copy()
            x["month_key"] = _month_key_from_date_series(x["date"])
            x["operational_data_flag"] = True
            x["financial_data_flag"] = False
            month_rows.append(x)

    if chem_path.exists():
        chem = pd.read_csv(chem_path, parse_dates=["date"])
        if not chem.empty:
            x = chem[["date"]].copy()
            x["month_key"] = _month_key_from_date_series(x["date"])
            x["operational_data_flag"] = True
            if "spend" in chem.columns:
                x["financial_data_flag"] = chem["spend"].notna().astype(bool).values
            else:
                x["financial_data_flag"] = False
            month_rows.append(x)

    if fail_path.exists():
        fail = pd.read_csv(fail_path, parse_dates=["fail_date"])
        if not fail.empty:
            x = fail[["fail_date"]].rename(columns={"fail_date": "date"}).copy()
            x["month_key"] = _month_key_from_date_series(x["date"])
            x["operational_data_flag"] = True
            x["financial_data_flag"] = False
            month_rows.append(x)

    if scada_path.exists():
        scada = pd.read_csv(scada_path, parse_dates=["date"])
        if not scada.empty:
            x = scada[["date"]].copy()
            x["month_key"] = _month_key_from_date_series(x["date"])
            x["operational_data_flag"] = True
            x["financial_data_flag"] = False
            month_rows.append(x)

    if not month_rows:
        logger.warning("No modeled facts found to build fact_month_status.")
        return pd.DataFrame()

    all_dates = pd.concat(month_rows, ignore_index=True)
    all_dates["date"] = pd.to_datetime(all_dates["date"], errors="coerce")

    summary = all_dates.groupby("month_key", as_index=False).agg(
        operational_data_through_date=("date", "max"),
        operational_records=("operational_data_flag", "sum"),
        financial_records=("financial_data_flag", "sum"),
    )

    today = pd.Timestamp.today().normalize()
    current_month_key = today.strftime("%Y-%m")
    previous_month_key = (today - pd.offsets.MonthBegin(1)).strftime("%Y-%m")

    def _status(month_key: str, fin_records: int) -> str:
        if month_key == current_month_key:
            return "OPEN"
        if month_key == previous_month_key and fin_records == 0:
            return "PRELIMINARY"
        return "FINAL"

    summary["financial_data_through_date"] = summary["operational_data_through_date"].where(
        summary["financial_records"] > 0, pd.NaT
    )
    summary["month_status"] = summary.apply(
        lambda r: _status(r["month_key"], r["financial_records"]),
        axis=1,
    )
    summary["last_refresh_timestamp"] = pd.Timestamp.now()
    summary["finalization_date"] = pd.NaT
    summary["finalized_by"] = None

    lock_df = _load_month_lock()
    if not lock_df.empty:
        summary = summary.merge(lock_df, how="left", on="month_key")
        summary["lock_flag"] = summary["lock_flag"].fillna(False)
    else:
        summary["lock_flag"] = False
        summary["lock_reason"] = None

    summary.loc[summary["lock_flag"] == True, "month_status"] = "FINAL"
    summary["operational_data_through_date"] = pd.to_datetime(
        summary["operational_data_through_date"], errors="coerce"
    ).dt.date
    summary["financial_data_through_date"] = pd.to_datetime(
        summary["financial_data_through_date"], errors="coerce"
    ).dt.date

    write_table(summary, modeled_dir, "fact_month_status", settings)
    batch.set_row_count("fact_month_status", len(summary))
    logger.info("Built fact_month_status | rows=%s", len(summary))
    return summary
