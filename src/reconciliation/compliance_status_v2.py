from __future__ import annotations

import pandas as pd

from src.common.paths import get_path


DEFAULT_TOLERANCE_PCT = 0.10


def load_chem_tolerance(settings=None) -> pd.DataFrame:
    """
    Loads optional chemistry tolerance configuration.

    Expected columns if file exists:
        chem_type
        tolerance_pct

    Returns empty DataFrame if no config is found.
    """
    try:
        if settings is None:
            return pd.DataFrame()

        modeled_dir = get_path(settings, "modeled")
        path = modeled_dir / "chem_tolerance.csv"

        if not path.exists():
            return pd.DataFrame()

        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()

        if "chem_type" in df.columns:
            df["chem_type"] = df["chem_type"].astype(str).str.strip().str.upper()

        if "tolerance_pct" in df.columns:
            df["tolerance_pct"] = pd.to_numeric(df["tolerance_pct"], errors="coerce")

        return df

    except Exception:
        return pd.DataFrame()


def _resolve_tolerance(chem_type, tol_df: pd.DataFrame | None) -> float:
    tol = DEFAULT_TOLERANCE_PCT

    if tol_df is None or tol_df.empty:
        return tol

    if "chem_type" not in tol_df.columns:
        return tol

    chem_type_norm = str(chem_type).strip().upper() if pd.notna(chem_type) else None
    if not chem_type_norm:
        return tol

    match = tol_df[tol_df["chem_type"].astype(str).str.strip().str.upper() == chem_type_norm]
    if match.empty:
        return tol

    if "tolerance_pct" in match.columns:
        value = pd.to_numeric(match.iloc[0]["tolerance_pct"], errors="coerce")
        if pd.notna(value):
            return float(value)

    if "tolerance" in match.columns:
        value = pd.to_numeric(match.iloc[0]["tolerance"], errors="coerce")
        if pd.notna(value):
            return float(value)

    return tol


def classify_operational_status(
    target_rate=None,
    actual_rate=None,
    chem_type=None,
    tol_df: pd.DataFrame | None = None,
    expected_rate=None,
    missing_target_flag=None,
    missing_actual_flag=None,
    no_production_flag=None,
    no_chemical_flag=None,
    stale_target_flag=None,
) -> str:
    """
    Chemistry operational status using upgraded engineering logic.
    Priority:
        MISSING_TARGET
        MISSING_ACTUAL
        NO_PRODUCTION
        NO_CHEMICAL
        STALE_TARGET
        UNDERFEED / OVERFEED / ON_TARGET
        UNKNOWN
    """
    def _flag(v) -> int:
        num = pd.to_numeric(v, errors="coerce")
        return int(num) if pd.notna(num) else 0

    if _flag(missing_target_flag) == 1:
        return "MISSING_TARGET"

    if _flag(missing_actual_flag) == 1:
        return "MISSING_ACTUAL"

    if _flag(no_production_flag) == 1:
        return "NO_PRODUCTION"

    if _flag(no_chemical_flag) == 1:
        return "NO_CHEMICAL"

    if _flag(stale_target_flag) == 1:
        return "STALE_TARGET"

    actual_rate = pd.to_numeric(actual_rate, errors="coerce")
    target_rate = pd.to_numeric(target_rate, errors="coerce")
    expected_rate = pd.to_numeric(expected_rate, errors="coerce")

    base_rate = expected_rate if pd.notna(expected_rate) and expected_rate != 0 else target_rate

    if pd.isna(actual_rate) or pd.isna(base_rate) or base_rate == 0:
        return "UNKNOWN"

    tol = _resolve_tolerance(chem_type=chem_type, tol_df=tol_df)
    variance_pct = (actual_rate - base_rate) / base_rate

    if pd.isna(variance_pct):
        return "UNKNOWN"
    if variance_pct < -tol:
        return "UNDERFEED"
    if variance_pct > tol:
        return "OVERFEED"
    return "ON_TARGET"