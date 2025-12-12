"""Engine for programmatically running the FLL Scheduler GA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config.app_config import AppConfig


logger = logging.getLogger(__name__)


def init_logging(app_config: AppConfig) -> None:
    """Initialize logging for the application."""
    logging_model = app_config.logging
    file = logging.FileHandler(
        filename=Path(logging_model.log_file),
        mode="w",
        encoding="utf-8",
        delay=True,
    )
    file.setLevel(logging_model.loglevel_file)
    file.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s] %(message)s"))

    console = logging.StreamHandler()
    console.setLevel(logging_model.loglevel_console)
    console.setFormatter(logging.Formatter("%(levelname)s[%(module)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file)
    root.addHandler(console)

    root.debug("Start: Tournament Scheduler.")
    app_config.log_creation_info()
