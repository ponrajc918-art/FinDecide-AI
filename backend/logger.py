"""Centralized logging with file rotation and structured formatting."""
import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
os.makedirs(LOG_DIR, exist_ok=True)

FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(ch)

    # Rotating file handler (10 MB × 5 backups)
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "findecide.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(FMT, DATE_FMT))
    logger.addHandler(fh)

    return logger
