"""Logging helpers for the agent."""

import logging
import os
from pathlib import Path
from typing import Optional


def _default_level() -> int:
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a module logger with a sane default configuration."""
    logger = logging.getLogger(name if name else "agent_test_gpt")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=_default_level(),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def setup_run_logging(output_dir: str, level: Optional[str] = None) -> logging.Logger:
    """Configure logging to emit to a per-run log file.

    Args:
        output_dir: directory where logs will be written.
        level: optional string level override (e.g., "DEBUG", "INFO").
    """
    log_level = getattr(logging, level.upper(), _default_level()) if level else _default_level()
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "agent.log"

    # Avoid duplicate handlers if called multiple times
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(logfile):
            root_logger.removeHandler(h)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return root_logger
