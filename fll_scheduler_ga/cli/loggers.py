"""Main cli api for the fll-scheduler-ga package."""

from __future__ import annotations

import json
import logging.config
from typing import TYPE_CHECKING

from ..config.constants import LOGGING_CONFIG_PATH

if TYPE_CHECKING:
    from pathlib import Path


def initialize_logging(path: Path | None = None) -> None:
    """Initialize logging for the application."""
    if path is None:
        path = LOGGING_CONFIG_PATH

    with path.open("r", encoding="utf-8") as f:
        logging_config_dict = json.load(f)
        logging.config.dictConfig(logging_config_dict)
