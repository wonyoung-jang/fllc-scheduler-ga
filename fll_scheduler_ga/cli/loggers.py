"""Main cli api for the fll-scheduler-ga package."""

from __future__ import annotations

import json
import logging.config

from ..config.constants import LOGGING_CONFIG_PATH


def initialize_logging() -> None:
    """Initialize logging for the application."""
    with LOGGING_CONFIG_PATH.open("r", encoding="utf-8") as f:
        logging_config = json.load(f)
        logging.config.dictConfig(logging_config)
