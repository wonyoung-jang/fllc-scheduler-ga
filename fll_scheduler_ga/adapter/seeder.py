"""Seed data I/O for genetic algorithm."""

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import DATA_MODEL_VERSION, SeedIslandStrategy, SeedPopSort

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fll_scheduler_ga.domain.model import TournamentConfig
    from fll_scheduler_ga.domain.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class GASeedData:
    """GA seed data object."""

    config: TournamentConfig | None = None
    population: list[Schedule] = field(default_factory=list)
    version: int = DATA_MODEL_VERSION


def distributed_seeding(seed_indices: Iterator[int], n_islands: int) -> dict[int, list[int]]:
    """Get the seed indices for each island."""
    island_to_seed: dict[int, list[int]] = defaultdict(list)
    for idx in seed_indices:
        island_to_seed[idx % n_islands].append(idx)
    return island_to_seed


def concentrated_seeding(seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
    """Get the seed indices for each island."""
    island_to_seed: dict[int, list[int]] = defaultdict(list)
    for i in range(n_islands):
        while len(island_to_seed[i]) < n_pop:
            if (idx := next(seed_indices, None)) is None:
                break
            island_to_seed[i].append(idx)
    return island_to_seed


@dataclass(slots=True)
class GASeeder:
    """Seeding strategies for GA instances."""

    rng: np.random.Generator
    seed_pop_size: int
    seed_island_strategy: str
    seed_pop_sort: str
    n_islands: int
    n_pop: int

    def is_valid(self) -> bool:
        """Check if seeding is valid based on the provided seed population."""
        if not self.seed_pop_size:
            logger.debug("No seed population provided. Starting with a fresh population.")
            return False
        logger.debug("Seeding population with %d individuals from seed file.", self.seed_pop_size)
        logger.debug("Seed pop sort: %s | Seed island strategy: %s", self.seed_pop_sort, self.seed_island_strategy)
        return True

    def get_island_seed_map(self) -> dict[int, list[int]]:
        """Get the mapping of islands to seed indices based on the seeding strategy."""
        if not self.is_valid():
            return {}
        match self.seed_island_strategy:
            case SeedIslandStrategy.CONCENTRATED:
                return concentrated_seeding(self.iter_seeds(), self.n_islands, self.n_pop)
            case SeedIslandStrategy.DISTRIBUTED:
                return distributed_seeding(self.iter_seeds(), self.n_islands)
            case _:
                return distributed_seeding(self.iter_seeds(), self.n_islands)

    def iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        match self.seed_pop_sort:
            case SeedPopSort.BEST:
                yield from np.arange(self.seed_pop_size)
            case SeedPopSort.RANDOM:
                yield from self.rng.permutation(self.seed_pop_size)
            case _:
                yield from self.rng.permutation(self.seed_pop_size)


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
