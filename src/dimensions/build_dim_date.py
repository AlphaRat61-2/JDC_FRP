from __future__ import annotations

import pandas as pd

from src.common.paths import get_path
from src.io.writers import write_table


def build_dim_date(settings, logger, batch) -> pd.DataFrame:
    staged_dir = get_path(settings, "staged")
    modeled_dir = get_path(settings, "modeled")

    prod_path = staged_dir / "stg_production_daily.csv"
    rates_path = staged_dir / "stg_chemical_rates.csv"
    fail_path = staged_dir / "stg_failures.csv"

    dates = []

    for path, col in [
        (prod_path, "date"),
        (rates_path, "date"),
        (fail_path, "install_date"),
        (fail_path, "fail_date"),
    ]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=[col])
            dates.extend(df[col].dropna().tolist())

    if not dates:
        logger.warning("No dates found to build dim_date.")
        return pd.DataFrame()

    min_date = min(dates)
    max_date = max(dates)
    dim = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})

    dim["year"] = dim["date"].dt.year
    dim["month_num"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%b")
    dim["month_key"] = dim["date"].dt.strftime("%Y-%m")
    dim["quarter"] = dim["date"].dt.quarter
    dim["day_of_month"] = dim["date"].dt.day
    dim["day_of_week"] = dim["date"].dt.day_name()
    dim["week_num"] = dim["date"].dt.isocalendar().week.astype(int)
    dim["is_month_end"] = dim["date"].dt.is_month_end
    dim["days_in_month"] = dim["date"].dt.days_in_month

    dim["date"] = dim["date"].dt.date

    write_table(dim, modeled_dir, "dim_date", settings)
    batch.set_row_count("dim_date", len(dim))
    logger.info("Built dim_date | rows=%s", len(dim))
    return dim