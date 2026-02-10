"""Shared configuration logging utilities.

This module provides helpers for consistent configuration logging across
train.py and test.py entry points to avoid code duplication.

Functions:
    - log_config_section: Log a configuration section with consistent
      formatting

Examples:
    >>> log_config_section("Training", {"epochs": 100, "lr": 0.001})

See Also:
    - src.train: Training entry point
    - src.test: Test entry point
"""

# Standard library imports
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_config_section(title: str, items: dict[str, Any]) -> None:
    """Log a configuration section with consistent formatting.

    Args:
        title: Section title to display.
        items: Dictionary of key-value pairs to log.
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    for key, value in items.items():
        logger.info("%s: %s", key, value)
    logger.info("=" * 60)
