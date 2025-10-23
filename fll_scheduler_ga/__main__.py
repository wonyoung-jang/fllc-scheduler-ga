"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import fll_scheduler_ga.config.cli as ga_cli
import fll_scheduler_ga.io.ga_exporter as ga_export
from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.config.ga_context import GaContext
from fll_scheduler_ga.genetic.ga import GA

if TYPE_CHECKING:
    from argparse import Namespace


logger = getLogger(__name__)


def setup_environment() -> tuple[Namespace, GA]:
    """Set up the environment for the application."""
    try:
        args = ga_cli.init_parser()
        ga_cli.init_logging(args)
        app_config = AppConfig.build(args, Path(args.config_file))
        ga_context = GaContext.build(args, app_config)
        ga = GA.build(args, ga_context)
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
        output_dir = getattr(args, "output_dir", "output")
        front_only = getattr(args, "front_only", False)
        cmap_name = getattr(args, "cmap_name", "viridis")
        no_plotting = getattr(args, "no_plotting", False)
        ga_export.generate_summary(
            ga=ga,
            output_dir=output_dir,
            cmap_name=cmap_name,
            front_only=front_only,
            no_plotting=no_plotting,
        )

    logger.debug("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
