import logging
import logging.handlers
from pathlib import Path

from src.common.paths import LOG_DIR


LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
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

    rotating_handler.setFormatter(formatter)
    rotating_handler.suffix = "%Y-%m-%d"

    logger.addHandler(rotating_handler)

    return logger


def set_debug_mode():

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)

def setup_logger(name="pipeline"):

    """
    Compatibility wrapper so older modules
    can call setup_logger().
    """

    return get_logger(name)