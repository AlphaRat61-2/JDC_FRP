from pathlib import Path
from datetime import datetime
import pandas as pd

from src.common.paths import ROOT


EXCEPTION_FILE = ROOT / "data" / "reports" / "exception_log.csv"


def _empty_exceptions():
    return pd.DataFrame(columns=[
        "timestamp",
        "module",
        "well_id",
        "exception_type",
        "exception_message",
        "severity"
    ])


def load_exceptions():
    if EXCEPTION_FILE.exists():
        return pd.read_csv(EXCEPTION_FILE)
    return _empty_exceptions()


def append_exceptions(exceptions):
    """
    Append a list of exception dictionaries to the exception log.
    """
    if exceptions is None:
        return load_exceptions()

    if isinstance(exceptions, dict):
        exceptions = [exceptions]

    if len(exceptions) == 0:
        return load_exceptions()

    df = load_exceptions()
    new_df = pd.DataFrame(exceptions)

    # normalize expected columns
    if "timestamp" not in new_df.columns:
        new_df["timestamp"] = datetime.utcnow().isoformat()

    if "module" not in new_df.columns:
        new_df["module"] = None

    if "well_id" not in new_df.columns:
        new_df["well_id"] = None

    if "exception_type" not in new_df.columns:
        new_df["exception_type"] = None

    if "exception_message" not in new_df.columns:
        if "message" in new_df.columns:
            new_df["exception_message"] = new_df["message"]
        else:
            new_df["exception_message"] = None

    if "severity" not in new_df.columns:
        new_df["severity"] = "ERROR"

    new_df = new_df[
        ["timestamp", "module", "well_id", "exception_type", "exception_message", "severity"]
    ].copy()

    df = pd.concat([df, new_df], ignore_index=True)

    EXCEPTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXCEPTION_FILE, index=False)

    return df


def store_exception(module, exception_type, message, well_id=None, severity="ERROR"):
    """
    Append a single exception row.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "module": module,
        "well_id": well_id,
        "exception_type": exception_type,
        "exception_message": message,
        "severity": severity
    }
    return append_exceptions([row])


def clear_exceptions():
    if EXCEPTION_FILE.exists():
        EXCEPTION_FILE.unlink()


def summarize_exceptions():
    df = load_exceptions()

    if df.empty:
        return df

    summary = (
        df.groupby(["exception_type", "severity"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return summary