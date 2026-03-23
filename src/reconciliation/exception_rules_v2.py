from __future__ import annotations

import pandas as pd


def assign_exception_code(row) -> str:
    """
    Consolidated chemistry exception coding aligned to upgraded recon/report logic.
    Highest-priority conditions first.
    """

    def _flag(name: str) -> int:
        value = row.get(name)
        num = pd.to_numeric(value, errors="coerce")
        return int(num) if pd.notna(num) else 0

    missing_target_flag = _flag("missing_target_flag")
    missing_actual_flag = _flag("missing_actual_flag")
    no_production_flag = _flag("no_production_flag")
    no_chemical_flag = _flag("no_chemical_flag")
    stale_target_flag = _flag("stale_target_flag")

    feed_status = str(row.get("feed_status", "")).strip().upper()
    operational_status = str(row.get("operational_status", "")).strip().upper()
    financial_status = str(row.get("financial_status", "")).strip().upper()

    if missing_target_flag == 1:
        return "CHEM_MISSING_TARGET"

    if missing_actual_flag == 1:
        return "CHEM_MISSING_ACTUAL"

    if no_production_flag == 1:
        return "CHEM_NO_PRODUCTION"

    if no_chemical_flag == 1:
        return "CHEM_NO_CHEMICAL"

    if stale_target_flag == 1:
        return "CHEM_STALE_TARGET"

    if feed_status == "UNDERFEED" or operational_status == "UNDERFEED":
        return "CHEM_UNDERFEED"

    if feed_status == "OVERFEED" or operational_status == "OVERFEED":
        return "CHEM_OVERFEED"

    if financial_status == "TARGET_NO_SPEND":
        return "CHEM_TARGET_NO_SPEND"

    if financial_status == "SPEND_NO_TARGET":
        return "CHEM_SPEND_NO_TARGET"

    if feed_status == "ON_TARGET" or operational_status == "ON_TARGET":
        return "CHEM_OK"

    return "CHEM_REVIEW"