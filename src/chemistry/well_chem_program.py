from __future__ import annotations

import pandas as pd

from src.common.exceptions import make_exception
from src.common.paths import project_root


def load_well_chem_program() -> pd.DataFrame:
    path = project_root() / "config" / "well_chem_program.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={"well_id": str, "chemical_key": str})
        required = {
            "well_id",
            "chemical_key",
            "program_start_date",
            "program_end_date",
            "required_flag",
            "default_target_basis",
            "target_source_preference",
            "actual_source_preference",
            "notes",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"well_chem_program.csv missing columns: {sorted(missing)}")

        df["program_start_date"] = pd.to_datetime(df["program_start_date"], errors="coerce")
        df["program_end_date"] = pd.to_datetime(df["program_end_date"], errors="coerce")
        df["required_flag"] = (
            df["required_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        )
        return df

    return pd.DataFrame(
        columns=[
            "well_id",
            "chemical_key",
            "program_start_date",
            "program_end_date",
            "required_flag",
            "default_target_basis",
            "target_source_preference",
            "actual_source_preference",
            "notes",
        ]
    )


def expand_well_chem_program_daily(
    program_df: pd.DataFrame,
    *,
    batch_id: str,
) -> tuple[pd.DataFrame, list[dict]]:
    if program_df.empty:
        return pd.DataFrame(
            columns=[
                "well_id",
                "date",
                "chemical_key",
                "required_flag",
                "default_target_basis",
                "target_source_preference",
                "actual_source_preference",
            ]
        ), []

    rows: list[pd.DataFrame] = []
    exceptions: list[dict] = []

    for _, row in program_df.iterrows():
        well_id = row.get("well_id")
        chemical_key = row.get("chemical_key")
        start = row.get("program_start_date")
        end = row.get("program_end_date")

        if pd.isna(start) or pd.isna(end) or end < start:
            exceptions.append(
                make_exception(
                    table_name="well_chem_program",
                    record_key=f"{well_id}|{chemical_key}",
                    exception_category="CHEM_PROGRAM",
                    exception_code="INVALID_PROGRAM_DATE_RANGE",
                    severity="CRITICAL",
                    message="Invalid start/end date in well chemical program.",
                    batch_id=batch_id,
                )
            )
            continue

        daily = pd.DataFrame(
            {
                "well_id": str(well_id),
                "date": pd.date_range(start, end, freq="D").date,
                "chemical_key": chemical_key,
                "required_flag": row.get("required_flag", True),
                "default_target_basis": row.get("default_target_basis"),
                "target_source_preference": row.get("target_source_preference"),
                "actual_source_preference": row.get("actual_source_preference"),
            }
        )
        rows.append(daily)

    if not rows:
        return pd.DataFrame(), exceptions

    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    return out, exceptions
