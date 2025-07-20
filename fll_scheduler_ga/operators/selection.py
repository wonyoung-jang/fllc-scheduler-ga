"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..config.config import TournamentConfig
from ..genetic.ga_parameters import GaParameters
from ..genetic.schedule import Population, Schedule

logger = logging.getLogger(__name__)


def build_selections(config: TournamentConfig, rng: Random, ga_params: GaParameters) -> Iterator["Selection"]:
    """Build and return a tuple of selection operators based on the configuration."""
    if "genetic.selection" not in config.parser:
        msg = "No selection configuration section '[genetic.selection]' found."
        raise ValueError(msg)

    variant_map = {
        "TournamentSelect": lambda r: TournamentSelect(r, tournament_size=ga_params.selection_size),
        "RandomSelect": lambda r: RandomSelect(r),
    }

    config_str = config.parser["genetic.selection"].get("selection_types", "")
    enabled_variants = [v.strip() for v in config_str.split(",") if v.strip()]

    if not enabled_variants:
        logger.warning("No selection types enabled in the configuration. Selection will not occur.")
        return

    for variant_name in enabled_variants:
        if variant_name not in variant_map:
            msg = f"Unknown selection type in config: '{variant_name}'"
            raise ValueError(msg)
        else:
            selection_factory = variant_map[variant_name]
            yield selection_factory(rng)


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    rng: Random

    @abstractmethod
    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation."""
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)


@dataclass(slots=True)
class Elitism(Selection):
    """Elitism for NSGA-II.

    Selects the best individuals for the next generation by taking whole fronts until the population size is met.
    """

    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select the new generation based on non-dominated sorting and crowding distance."""
        yield from sorted(population, key=lambda p: (p.rank, -p.crowding))[:population_size]


@dataclass(slots=True)
class TournamentSelect(Selection):
    """Tournament selection for multi-objective problems using NSGA-II principles.

    Selects the winner based on rank, then crowding distance.
    """

    tournament_size: int

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using NSGA-II tournament selection."""
        yield from sorted(
            self.rng.sample(population, k=min(self.tournament_size, len(population))),
            key=lambda p: (p.rank, -p.crowding),
        )[:num_parents]


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals randomly from the population."""
        yield from self.rng.sample(population, k=num_parents)
