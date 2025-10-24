"""Engine for programmatically running the FLL Scheduler GA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fll_scheduler_ga.io.ga_exporter as ga_export
from fll_scheduler_ga.config.ga_context import GaContext
from fll_scheduler_ga.genetic.ga import GA

if TYPE_CHECKING:
    from .config.app_config import AppConfig


logger = logging.getLogger(__name__)


def run_ga_instance(app_config: AppConfig) -> None:
    """Initialize and run a complete GA instance from an AppConfig object.

    This is the primary entry point for programmatic execution, such as from a web API.

    Args:
        app_config (AppConfig): A fully constituted application configuration object.

    """
    try:
        ga_context = GaContext.build(app_config)
        ga = GA.build(ga_context)
    except Exception:
        logger.exception("Error building GA context from app_config")
        raise

    try:
        ga.run()
    except Exception:
        logger.exception("An unhandled error occurred during the GA run.")
    finally:
        args = app_config.arguments
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ga_export.generate_summary(
            ga=ga,
            output_dir=output_dir,
            cmap_name=args.cmap_name,
            front_only=args.front_only,
            no_plotting=args.no_plotting,
        )
    logger.debug("FLLC Scheduler run finished for output: %s", output_dir)
