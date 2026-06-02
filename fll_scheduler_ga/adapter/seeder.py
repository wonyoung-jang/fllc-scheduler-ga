"""Seed data I/O for genetic algorithm."""

import pickle
from logging import getLogger
from typing import TYPE_CHECKING

from fll_scheduler_ga.constants import DATA_MODEL_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from fll_scheduler_ga.domain.model import BenchmarkSeedData, GASeedData, Schedule, TournamentConfig

logger = getLogger(__name__)


def load_ga(path: Path, config: TournamentConfig) -> list[Schedule]:
    """Load GA seed data from a file."""
    try:
        logger.debug("Loading seed population from: %s", path)
        with path.open("rb") as f:
            data: GASeedData = pickle.load(f)
    except OSError, pickle.PicklingError, AttributeError, ModuleNotFoundError:
        logger.warning("Could not load or parse seed file. Starting with a fresh population.")
        return []
    except EOFError:
        logger.debug("Pickle file is empty")
        return []
    try:
        pop = []
        if data.version != DATA_MODEL_VERSION:
            logger.warning(
                "Seed population data version mismatch: Expected (%d), found (%d). Dismissing old seed file...",
                DATA_MODEL_VERSION,
                data.version,
            )
        elif data.config != config:
            logger.warning("Seed population does not match current config. Using current...")
            data.config = config
        elif not data.population:
            logger.warning("Seed population is missing. Using current...")
        else:
            pop = data.population
    except AttributeError:
        logger.warning("Seed population is malformed. Starting with a fresh population.")
        return []
    data.population = pop
    return data.population


def save_ga(path: Path, data: GASeedData) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    try:
        logger.debug("Saving final population of size %d to seed file: %s", len(data.population), path)
        with path.open("wb") as f:
            pickle.dump(data, f)
    except OSError, pickle.PicklingError, EOFError:
        logger.exception("Error saving population to seed file: %s", path)


def load_fitness_benchmark(path: Path) -> BenchmarkSeedData | None:
    """Load benchmark data from a pickle file."""
    logger.debug("Loading fitness benchmarks from cache: %s", path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            seed_data = pickle.load(f)
    except OSError, EOFError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError:
        logger.debug("Failed to load fitness benchmarks from cache: %s", path)
        return None
    return seed_data


def save_fitness_benchmark(path: Path, data: BenchmarkSeedData) -> None:
    """Save benchmark data to a pickle file."""
    try:
        with path.open("wb") as f:
            pickle.dump(data, f)
        logger.info("Fitness benchmarks saved to cache: %s", path)
    except OSError, pickle.PicklingError, EOFError:
        logger.exception("Failed to save fitness benchmarks to cache.")
