"""Application logging setup using Python's standard logging."""
import logging
import sys
from .config import get_config


def setup_logging(level: str | None = None) -> None:
    """Configure le logging standard."""
    config = get_config()
    log_level_name = level or config["logging"]["level"]
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(handler)
    root.setLevel(log_level)


logger = logging.getLogger("anomaly_detection")
