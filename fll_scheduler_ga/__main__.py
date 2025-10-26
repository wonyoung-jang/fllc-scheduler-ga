"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

import logging

from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.config.constants import CONFIG_FILE
from fll_scheduler_ga.engine import init_logging, run_ga_instance

logger = logging.getLogger(__name__)


def main_cli() -> None:
    """Run the fll-scheduler-ga application from the command line interface."""
    try:
        app_config = AppConfig.build(CONFIG_FILE)
        init_logging(app_config)
        run_ga_instance(app_config)
    except Exception:
        logger.exception("GA process failed unexpectedly.")
        raise


if __name__ == "__main__":
    main_cli()
