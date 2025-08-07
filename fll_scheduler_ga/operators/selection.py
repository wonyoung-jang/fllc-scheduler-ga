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

from ..config.app_config import AppConfig
from ..config.constants import SelectionOp
from ..data_model.schedule import Population, Schedule

logger = logging.getLogger(__name__)


def build_selections(app_config: AppConfig) -> Iterator["Selection"]:
    """Build and return a tuple of selection operators based on the configuration."""
    variant_map = {
        SelectionOp.TOURNAMENT_SELECT: lambda: TournamentSelect(app_config.rng, app_config.ga_params.selection_size),
        SelectionOp.RANDOM_SELECT: lambda: RandomSelect(app_config.rng),
    }

    if not app_config.operators.selection_types:
        logger.warning("No selection types enabled in the configuration. Selection will not occur.")
        return

    for variant_name in app_config.operators.selection_types:
        if variant_name not in variant_map:
            msg = f"Unknown selection type in config: '{variant_name}'"
            raise ValueError(msg)
        else:
            selection_factory = variant_map[variant_name]
            yield selection_factory()


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    rng: Random

    @abstractmethod
    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation."""
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)


@dataclass(slots=True)
class TournamentSelect(Selection):
    """Tournament selection for multi-objective problems using NSGA-III principles."""

    tournament_size: int

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using NSGA-III tournament selection."""
        contenders = self.rng.sample(population, k=self.tournament_size)
        tournament = sorted(contenders, key=lambda p: (p.rank, self.rng.choice([True, False])))
        yield from tournament[:num_parents]


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals randomly from the population."""
        yield from self.rng.sample(population, k=num_parents)
