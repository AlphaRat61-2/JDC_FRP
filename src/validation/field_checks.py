from __future__ import annotations

import pandas as pd


def validate_not_null(df: pd.DataFrame, col: str, table_name: str) -> list[dict]:
    bad = df[df[col].isna()]
    return [
        {
            "table_name": table_name,
            "record_key": str(idx),
            "exception_category": "FIELD",
            "exception_code": "NULL_REQUIRED_FIELD",
            "severity": "WARNING",
            "message": f"Null value found in required field: {col}",
        }
        for idx in bad.index
    ]


def parse_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def validate_numeric_nonnegative(df: pd.DataFrame, col: str, table_name: str) -> list[dict]:
    bad = df[df[col].fillna(0) < 0]
    return [
        {
            "table_name": table_name,
            "record_key": str(idx),
            "exception_category": "FIELD",
            "exception_code": "NEGATIVE_VALUE",
            "severity": "WARNING",
            "message": f"Negative value found in field: {col}",
        }
        for idx in bad.index
    ]