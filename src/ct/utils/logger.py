import logging
import sys
from pathlib import Path
from typing import Optional

"""
src/utils/logger.py
--------------------------------
This module provides a utility function to create and configure loggers.
"""

def configure_logging(
    log_file: str,
    level: int = logging.INFO,
    force: bool = False
) -> None:
    """
    Configure root logger with a stream handler (console)
    and a file handler (log_file). Safe to call multiple times;
    only the first call actually adds handlers.
    """
    root = logging.getLogger()

    # If handlers already exist, don't reconfigure
    if root.hasHandlers() and not force:
        return
    if force:
        root.handlers.clear()

    root.setLevel(level)

    # Ensure log file is not None
    if log_file is None:
        raise ValueError("log_file must be specified for file logging.")
    # Ensure log directory exists
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Console handler
    stream_handler = logging.StreamHandler()
    # File handler
    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def log_unhandled(exc_type, exc, tb) -> None:
    """
    Global hook for unhandled exceptions.
    Logs them as CRITICAL, then lets the process die.
    Intended to be assigned to sys.excepthook in main.
    """
    logger = logging.getLogger(__name__)
    # Log full traceback
    logger.critical("UNHANDLED EXCEPTION", exc_info=(exc_type, exc, tb))


def install_excepthook() -> None:
    """
    Convenience helper to install log_unhandled as the global excepthook.
    Call this once in your main script after configure_logging().
    """
    sys.excepthook = log_unhandled


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Thin wrapper around logging.getLogger for convenience.
    If name is None, returns a logger for this module.
    """
    if name is None:
        name = __name__
    return logging.getLogger(name)