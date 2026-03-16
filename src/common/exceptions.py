from datetime import datetime


def make_exception(module, message, well_id=None, severity="ERROR"):
    """
    Standard exception record used across the platform.
    """

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "module": module,
        "well_id": well_id,
        "severity": severity,
        "message": message
    }