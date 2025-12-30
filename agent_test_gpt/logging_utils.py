"""Logging helpers for the agent."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a module logger with a sane default configuration."""
    logger = logging.getLogger(name if name else "agent_test_gpt")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger
