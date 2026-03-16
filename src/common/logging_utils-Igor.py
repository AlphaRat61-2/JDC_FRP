import logging
import logging.handlers
from pathlib import Path

from src.common.paths import LOG_DIR


LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "platform", level: str = "INFO"):
    logger = logging.getLogger(str(name))

    if logger.handlers:
        return logger

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = LOG_DIR / "platform.log"

    rotating_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    rotating_handler.setLevel(numeric_level)
    rotating_handler.setFormatter(formatter)
    rotating_handler.suffix = "%Y-%m-%d"
    logger.addHandler(rotating_handler)

    logger.propagate = False
    return logger


def setup_logger(settings=None):
    """
    Compatibility wrapper.

    Accepts either:
    - a settings dict
    - a logger name string
    - None
    """
    if isinstance(settings, dict):
        name = settings.get("project_name") or settings.get("system", {}).get("name") or "platform"

        level = (
            settings.get("logging", {}).get("level")
            or settings.get("log_level")
            or "INFO"
        )
        return get_logger(name=name, level=level)

    if isinstance(settings, str):
        return get_logger(name=settings)

    return get_logger()


def set_debug_mode():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)