from __future__ import annotations

import pandas as pd


def classify_financial_status(
    target_exists=False,
    spend_exists=False,
    actual_exists=None,
    missing_target_flag=None,
    missing_actual_flag=None,
) -> str:
    """
    Financial/data-presence style classification for chemistry.
    """
    def _boolish(v) -> bool:
        if isinstance(v, bool):
            return v
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num):
            return bool(int(num))
        return False

    target_exists = _boolish(target_exists)
    spend_exists = _boolish(spend_exists)
    actual_exists = _boolish(actual_exists)

    if pd.notna(pd.to_numeric(missing_target_flag, errors="coerce")):
        target_exists = not bool(int(pd.to_numeric(missing_target_flag, errors="coerce")))

    if pd.notna(pd.to_numeric(missing_actual_flag, errors="coerce")):
        actual_exists = not bool(int(pd.to_numeric(missing_actual_flag, errors="coerce")))

    if target_exists and (spend_exists or actual_exists):
        return "ALIGNED"

    if target_exists and not (spend_exists or actual_exists):
        return "TARGET_NO_SPEND"

    if not target_exists and (spend_exists or actual_exists):
        return "SPEND_NO_TARGET"

    return "NO_TARGET_NO_SPEND"