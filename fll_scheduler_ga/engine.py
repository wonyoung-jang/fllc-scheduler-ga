"""Engine for programmatically running the FLL Scheduler GA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fll_scheduler_ga.io.ga_exporter as ga_export
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.genetic.ga_context import GaContext

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


def run_ga_instance(app_config: AppConfig) -> None:
    """Initialize and run a complete GA instance from an AppConfig object.

    This is the primary entry point for programmatic execution, such as from a web API.

    Args:
        app_config (AppConfig): A fully constituted application configuration object.

    """
    try:
        context = GaContext.build(app_config)
        ga = GA.build(context)
    except Exception:
        logger.exception("Error building GA context from app_config")
        raise

    try:
        ga.run()
    except Exception:
        logger.exception("An unhandled error occurred during the GA run.")
    finally:
        exports = app_config.exports
        output_dir = Path(exports.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ga_export.generate_summary(
            ga=ga,
            output_dir=output_dir,
            export_model=exports,
        )
    logger.debug("FLLC Scheduler run finished for output: %s", output_dir)
