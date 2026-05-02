from __future__ import annotations

import logging
import os
import sys


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_level: str | None = None) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level:
            Logging level string. If None, reads LOG_LEVEL from environment.
    """
    resolved_level = (log_level or os.getenv("LOG_LEVEL") or "INFO").upper()

    numeric_level = getattr(logging, resolved_level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name:
            Logger name, usually __name__.

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)