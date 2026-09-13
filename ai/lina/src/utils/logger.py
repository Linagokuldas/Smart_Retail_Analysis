"""
Logger utility for the Lina Analytics Module.

Sets up a consistent, structured logger that can be used across all submodules.
Outputs to both console and file.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name (typically the module __name__).
        log_level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (logs go into ai/lina/outputs/reports/)
    log_dir = _resolve_log_dir()
    if log_dir:
        log_file = log_dir / f"lina_{datetime.now().strftime('%Y%m%d')}.log"
        try:
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # If file logging fails (e.g., read-only filesystem), silently continue
            pass

    return logger


def _resolve_log_dir() -> Path | None:
    """
    Resolve the log directory relative to the module root.

    Returns:
        Path to the log directory, or None if it cannot be resolved.
    """
    try:
        # Walk up from this file to find the 'ai/lina' root
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "outputs" / "reports"
            if candidate.exists():
                return candidate
        return None
    except Exception:
        return None
