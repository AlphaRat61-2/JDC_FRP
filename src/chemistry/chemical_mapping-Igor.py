from __future__ import annotations

import pandas as pd

from src.common.exceptions import make_exception
from src.common.paths import project_root


def load_chem_name_map() -> pd.DataFrame:
    path = project_root() / "config" / "chem_name_map.csv"
    if path.exists():
        df = pd.read_csv(path)
        required = {
            "chem_name",
            "chemical_key",
            "normalized_chemical_name",
            "chem_type",
            "dose_basis",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"chem_name_map.csv missing columns: {sorted(missing)}")
        return df
    return pd.DataFrame(
        columns=[
            "chem_name",
            "chemical_key",
            "normalized_chemical_name",
            "chem_type",
            "dose_basis",
        ]
    )


def load_basis_rules() -> pd.DataFrame:
    path = project_root() / "config" / "chem_basis_rules.csv"
    if path.exists():
        df = pd.read_csv(path)
        required = {"chemical_key", "default_basis", "allowed_basis_override_flag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"chem_basis_rules.csv missing columns: {sorted(missing)}")
        return df
    return pd.DataFrame(
        columns=["chemical_key", "default_basis", "allowed_basis_override_flag"]
    )


def map_chemical_names(
    df: pd.DataFrame,
    *,
    chem_name_col: str,
    chem_type_col: str | None,
    table_name: str,
    batch_id: str,
) -> tuple[pd.DataFrame, list[dict]]:
    chem_map = load_chem_name_map()
    work = df.copy()

    if chem_name_col not in work.columns:
        return work, []

    work[chem_name_col] = work[chem_name_col].astype(str).str.strip()
    chem_map["chem_name"] = chem_map["chem_name"].astype(str).str.strip()

    mapped = work.merge(
        chem_map,
        how="left",
        left_on=chem_name_col,
        right_on="chem_name",
        suffixes=("", "_map"),
    )

    exceptions: list[dict] = []
    bad = mapped[mapped["chemical_key"].isna()]

    for idx, row in bad.iterrows():
        record_key = f"{row.get('well_id', '')}|{row.get('date', '')}|{row.get(chem_name_col, '')}|{idx}"
        exceptions.append(
            make_exception(
                module=table_name,
                message=f"Chemical name not mapped: {row.get(chem_name_col, '')}",
                well_id=row.get("well_id"),
                severity="WARNING",
            )
        )

    if chem_type_col and chem_type_col in mapped.columns and "chem_type_map" in mapped.columns:
        mapped[chem_type_col] = mapped[chem_type_col].fillna(mapped["chem_type_map"])

    return mapped, exceptions
