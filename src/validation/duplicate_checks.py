from __future__ import annotations

import pandas as pd


def check_duplicates(df: pd.DataFrame, keys: list[str], table_name: str) -> list[dict]:
    dupes = df[df.duplicated(subset=keys, keep=False)]
    return [
        {
            "table_name": table_name,
            "record_key": "|".join(str(row[k]) for k in keys),
            "exception_category": "DUPLICATE",
            "exception_code": "DUPLICATE_BUSINESS_KEY",
            "severity": "WARNING",
            "message": f"Duplicate key detected on {keys}",
        }
        for _, row in dupes.iterrows()
    ]