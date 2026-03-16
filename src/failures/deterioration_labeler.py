import pandas as pd

def add_deterioration_features(scada_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Adds deterioration_score and pre_failure_flag to SCADA daily data.
    Placeholder logic for now — replace with real model later.
    """

    out = scada_daily.copy()

    # Placeholder deterioration score: 0 for all wells
    if "deterioration_score" not in out.columns:
        out["deterioration_score"] = 0.0

    # Placeholder pre-failure flag: 0 for all wells
    if "pre_failure_flag" not in out.columns:
        out["pre_failure_flag"] = 0

    return out