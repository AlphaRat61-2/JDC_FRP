from __future__ import annotations

import pandas as pd


def assign_exception_code(row: pd.Series) -> str:
    expected_flag = bool(row.get("expected_flag"))
    target_exists = bool(row.get("target_exists_flag"))
    actual_exists = bool(row.get("actual_exists_flag"))
    spend_exists = bool(row.get("spend_exists_flag"))
    chemical_key = row.get("chemical_key")
    basis = row.get("target_basis")
    target_ppm = row.get("target_ppm")

    if pd.isna(chemical_key):
        return "UNMAPPED_CHEMICAL"
    if pd.isna(basis):
        return "MISSING_BASIS"
    if expected_flag and not target_exists and spend_exists:
        return "EXPECTED_SPEND_NO_TARGET"
    if expected_flag and not target_exists:
        return "EXPECTED_NO_TARGET"
    if not expected_flag and target_exists:
        return "TARGET_NOT_IN_PROGRAM"
    if pd.isna(target_ppm):
        return "MISSING_PRODUCTION"
    if target_exists and not actual_exists and not spend_exists:
        return "TARGET_ONLY"
    if target_exists and not actual_exists and spend_exists:
        return "TARGET_AND_SPEND_NO_ACTUAL"
    if not target_exists and not actual_exists and spend_exists:
        return "SPEND_ONLY"
    if not target_exists and actual_exists:
        return "ACTUAL_NO_TARGET"
    if target_exists and actual_exists and spend_exists:
        return "FULL_MATCH"
    return "UNKNOWN"
