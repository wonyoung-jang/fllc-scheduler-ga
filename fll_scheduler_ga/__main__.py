"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.config.cli import init_logging, init_parser
from fll_scheduler_ga.io.export import generate_summary

if TYPE_CHECKING:
    from argparse import Namespace

    from fll_scheduler_ga.genetic.ga import GA


logger = getLogger(__name__)


def setup_environment() -> tuple[Namespace, GA]:
    """Set up the environment for the application."""
    try:
        args = init_parser()
        init_logging(args)
        app_config = AppConfig.create_app_config(args, Path(args.config_file))
        ga_context = app_config.create_ga_context()
        ga = ga_context.create_ga_instance(args)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return args, ga


def main() -> None:
    """Run the fll-scheduler-ga application."""
    env = setup_environment()
    if not env:
        logger.error("Failed to set up the environment. Exiting.")
        return

    args, ga = env

    try:
        ga.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Saving final population and results before exiting.")
    except Exception:
        logger.exception("An unhandled error occurred during the GA run. Saving state before exiting.")
    finally:
        generate_summary(args, ga)

    logger.debug("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
