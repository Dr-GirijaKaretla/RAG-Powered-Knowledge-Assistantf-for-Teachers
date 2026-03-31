"""
Centralized logging setup for the Teacher RAG application.

Provides a consistent, formatted logger instance for every module
in the project via the ``setup_logger`` factory function.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s]: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Keep a registry so each named logger is configured exactly once.
_configured_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    level: Optional[int] = logging.INFO,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for *name*.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Logging level (default ``INFO``).

    Returns
    -------
    logging.Logger
        A logger that writes to *stderr* with a uniform format:
        ``[TIMESTAMP] [LEVEL] [MODULE]: message``
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if the root logger already has one.
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent messages from bubbling to the root logger twice.
    logger.propagate = False

    _configured_loggers[name] = logger
    return logger
