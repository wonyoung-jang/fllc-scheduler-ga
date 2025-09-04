"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import SelectionOp

if TYPE_CHECKING:
    from collections.abc import Iterator
    from random import Random

    from ..config.app_config import AppConfig
    from ..data_model.schedule import Population, Schedule

logger = getLogger(__name__)


def build_selections(app_config: AppConfig) -> Iterator[Selection]:
    """Build and return a tuple of selection operators based on the configuration."""
    variant_map = {
        SelectionOp.TOURNAMENT_SELECT: lambda: TournamentSelect(
            app_config.rng,
            app_config.ga_params.selection_size,
        ),
        SelectionOp.RANDOM_SELECT: lambda: RandomSelect(
            app_config.rng,
        ),
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
    def select(self, population: Population, parents: int = 2) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation.

        Args:
            population (Population): The current population of schedules.
            parents (int): The number of parents to select.

        Yields:
            Schedule: The selected parent schedules.

        """


@dataclass(slots=True)
class TournamentSelect(Selection):
    """Tournament selection for multi-objective problems using NSGA-III principles."""

    tournament_size: int

    def select(self, population: Population, parents: int = 2) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation."""
        tournament = sorted(
            self.rng.sample(population, k=self.tournament_size),
            key=lambda p: (p.rank, -sum(p.fitness)),
        )
        yield from tournament[:parents]


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def select(self, population: Population, parents: int = 2) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation."""
        yield from self.rng.sample(population, k=parents)
