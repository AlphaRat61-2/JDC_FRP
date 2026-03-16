from __future__ import annotations

import pandas as pd


def validate_required_columns(df: pd.DataFrame, schema: dict, table_name: str) -> list[dict]:
    missing = [c for c in schema["required"] if c not in df.columns]
    exceptions = []
    for col in missing:
        exceptions.append(
            {
                "table_name": table_name,
                "record_key": "",
                "exception_category": "SCHEMA",
                "exception_code": "MISSING_REQUIRED_COLUMN",
                "severity": "CRITICAL",
                "message": f"Missing required column: {col}",
            }
        )
    return exceptions