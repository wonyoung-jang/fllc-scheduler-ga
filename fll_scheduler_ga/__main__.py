"""Main entry point for the fll-scheduler-ga package."""

from __future__ import annotations

import pickle
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from fll_scheduler_ga.config.app_config import create_app_config
from fll_scheduler_ga.config.cli import create_parser
from fll_scheduler_ga.io.export import generate_summary

if TYPE_CHECKING:
    from argparse import Namespace

    from fll_scheduler_ga.genetic.ga import GA


logger = getLogger(__name__)


def setup_environment() -> tuple[Namespace, GA]:
    """Set up the environment for the application."""
    try:
        args = create_parser()
        initialize_logging(args)
        app_config = create_app_config(args)
        ga_context = app_config.create_ga_context()
        ga = ga_context.create_ga_instance(args)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return args, ga


def initialize_logging(args: Namespace) -> None:
    """Initialize logging for the application."""
    file_handler = FileHandler(
        filename=args.log_file,
        mode="w",
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(args.loglevel_file)
    file_handler.setFormatter(Formatter("[%(asctime)s] %(levelname)s[%(module)s] %(message)s"))

    console_handler = StreamHandler()
    console_handler.setLevel(args.loglevel_console)
    console_handler.setFormatter(Formatter("%(levelname)s[%(module)s] %(message)s"))

    root_logger = getLogger()
    root_logger.setLevel(DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    args_str = "\n\t".join(f"  {k}: {v}" for k, v in args.__dict__.items())
    logger.debug("Starting FLLC Scheduler with args:\n\targparse.Namespace\n\t%s", args_str)


def save_population_to_seed_file(ga: GA, seed_file: str | Path, *, front: bool = False) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    population = ga.pareto_front() if front else ga.total_population

    if not population:
        logger.warning("No population to save to seed file.")
        return

    data_to_cache = {
        "population": population,
        "config": ga.context.app_config.tournament,
    }

    path = Path(seed_file)
    logger.debug("Saving final population of size %d to seed file: %s", len(population), path)

    try:
        with path.open("wb") as f:
            pickle.dump(data_to_cache, f)
    except (OSError, pickle.PicklingError, EOFError):
        logger.exception("Error saving population to seed file: %s", path)


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
        if ga.total_population:  # only save if a final population exists
            save_population_to_seed_file(ga, args.seed_file, front=args.front_only)
            generate_summary(args, ga)

    logger.debug("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
