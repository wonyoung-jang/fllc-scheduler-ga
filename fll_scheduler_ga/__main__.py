"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

import logging

import fll_scheduler_ga.io.ga_exporter as ga_export
from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.config.constants import (
    CMAP_NAME,
    CONFIG_FILE,
    FRONT_ONLY,
    LOG_FILE,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
    NO_PLOTTING,
    OUTPUT_DIR,
)
from fll_scheduler_ga.config.ga_context import GaContext
from fll_scheduler_ga.genetic.ga import GA

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

    root.debug("Start: FLLC Scheduler.")


def setup_environment() -> GA | None:
    """Set up the environment for the application."""
    try:
        init_logging()
        app_config = AppConfig.build(CONFIG_FILE)
        ga_context = GaContext.build(app_config)
        ga = GA.build(ga_context)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
        return None
    else:
        return ga


def main() -> None:
    """Run the fll-scheduler-ga application."""
    ga = setup_environment()
    if ga is None:
        logger.error("Failed to set up the environment. Exiting.")
        return

    try:
        ga.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Saving final population and results before exiting.")
    except Exception:
        logger.exception("An unhandled error occurred during the GA run. Saving state before exiting.")
    finally:
        ga_export.generate_summary(
            ga=ga,
            output_dir=OUTPUT_DIR,
            cmap_name=CMAP_NAME,
            front_only=FRONT_ONLY,
            no_plotting=NO_PLOTTING,
        )

    logger.debug("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
