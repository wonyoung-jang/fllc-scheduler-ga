"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

import logging

from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.config.constants import (
    CONFIG_FILE,
    LOG_FILE,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
)
from fll_scheduler_ga.engine import run_ga_instance

logger = logging.getLogger(__name__)


def init_logging() -> None:
    """Initialize logging for the application."""
    file = logging.FileHandler(
        filename=LOG_FILE,
        mode="w",
        encoding="utf-8",
        delay=True,
    )
    file.setLevel(LOG_LEVEL_FILE)
    file.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s] %(message)s"))

    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL_CONSOLE)
    console.setFormatter(logging.Formatter("%(levelname)s[%(module)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file)
    root.addHandler(console)

    root.debug("Start: Tournament Scheduler.")


def main_cli() -> None:
    """Run the fll-scheduler-ga application from the command line interface."""
    try:
        init_logging()
        app_config = AppConfig.build(CONFIG_FILE)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
        return

    run_ga_instance(app_config)


if __name__ == "__main__":
    main_cli()
