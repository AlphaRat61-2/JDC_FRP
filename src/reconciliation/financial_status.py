from __future__ import annotations


def classify_financial_status(*, target_exists: bool, spend_exists: bool) -> str:
    if not target_exists and spend_exists:
        return "SPEND_WITHOUT_TARGET"
    if target_exists and not spend_exists:
        return "TARGET_NO_SPEND"
    if target_exists and spend_exists:
        return "SPEND_PRESENT"
    return "UNKNOWN"
