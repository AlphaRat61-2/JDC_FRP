from pathlib import Path
from datetime import datetime
import pandas as pd

from src.common.paths import ROOT


EXCEPTION_FILE = ROOT / "data" / "reports" / "exception_log.csv"


def load_exceptions():

    if EXCEPTION_FILE.exists():
        return pd.read_csv(EXCEPTION_FILE)

    return pd.DataFrame(columns=[
        "timestamp",
        "module",
        "well_id",
        "exception_type",
        "exception_message",
        "severity"
    ])


def store_exception(module, exception_type, message, well_id=None, severity="ERROR"):

    df = load_exceptions()

    new_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "module": module,
        "well_id": well_id,
        "exception_type": exception_type,
        "exception_message": message,
        "severity": severity
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    EXCEPTION_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(EXCEPTION_FILE, index=False)


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