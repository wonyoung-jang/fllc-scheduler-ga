"""Seed data I/O for genetic algorithm."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import DATA_MODEL_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..data_model.app_schemas import TournamentConfig
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class GASeedData:
    """GA seed data object."""

    version: int = DATA_MODEL_VERSION
    config: TournamentConfig | None = None
    population: list[Schedule] = field(default_factory=list)


@dataclass(slots=True)
class GALoad:
    """Loader for GA instances from seed files."""

    seed_file: Path
    config: TournamentConfig

    def load(self) -> GASeedData | None:
        """Load and integrate a population from a seed file."""
        try:
            logger.debug("Loading seed population from: %s", self.seed_file)
            with self.seed_file.open("rb") as f:
                data: GASeedData = pickle.load(f)
        except (OSError, pickle.PicklingError, AttributeError, ModuleNotFoundError):
            logger.warning("Could not load or parse seed file. Starting with a fresh population.")
            return None
        except EOFError:
            logger.debug("Pickle file is empty")
            return None

        try:
            pop = []
            if data.version != DATA_MODEL_VERSION:
                logger.warning(
                    "Seed population data version mismatch: Expected (%d), found (%d). Dismissing old seed file...",
                    DATA_MODEL_VERSION,
                    data.version,
                )
            elif data.config != self.config:
                logger.warning("Seed population does not match current config. Using current...")
                data.config = self.config
            elif not data.population:
                logger.warning("Seed population is missing. Using current...")
            else:
                pop = data.population
        except AttributeError:
            logger.warning("Seed population is malformed. Starting with a fresh population.")
            return None

        data.population = pop
        return data


@dataclass(slots=True)
class GASave:
    """Saver for GA instances to seed files."""

    seed_file: Path
    data: GASeedData

    def save(self) -> None:
        """Save the final population to a file to be used as a seed for a future run."""
        try:
            seed_ga_data = self.data
            path = self.seed_file
            logger.debug("Saving final population of size %d to seed file: %s", len(seed_ga_data.population), path)
            with path.open("wb") as f:
                pickle.dump(seed_ga_data, f)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Error saving population to seed file: %s", path)


@dataclass(slots=True)
class SeedingStrategy(ABC):
    """Abstract base class for GA seeding strategies."""

    @abstractmethod
    def get_indices(self, seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
        """Get the seed indices for each island."""


@dataclass(slots=True)
class DistributedSeedingStrategy(SeedingStrategy):
    """Distributed seeding strategy for GA islands."""

    def get_indices(self, seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        island_to_seed: dict[int, list[int]] = defaultdict(list)
        for idx in seed_indices:
            island_to_seed[idx % n_islands].append(idx)
        return island_to_seed


@dataclass(slots=True)
class ConcentratedSeedingStrategy(SeedingStrategy):
    """Concentrated seeding strategy for GA islands."""

    def get_indices(self, seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        island_to_seed: dict[int, list[int]] = defaultdict(list)
        for i in range(n_islands):
            while len(island_to_seed[i]) < n_pop:
                if (idx := next(seed_indices, None)) is None:
                    break
                island_to_seed[i].append(idx)
        return island_to_seed
