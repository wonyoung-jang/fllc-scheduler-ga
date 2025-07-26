"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from random import Random

from ..config.config import OperatorConfig
from ..genetic.ga_parameters import GaParameters
from ..genetic.schedule import Population, Schedule

logger = logging.getLogger(__name__)


class SelectionKeys(StrEnum):
    """Enum for selection operator keys."""

    TOURNAMENT_SELECT = "TournamentSelect"
    RANDOM_SELECT = "RandomSelect"


def build_selections(o_config: OperatorConfig, rng: Random, ga_params: GaParameters) -> Iterator["Selection"]:
    """Build and return a tuple of selection operators based on the configuration."""
    variant_map = {
        SelectionKeys.TOURNAMENT_SELECT: lambda: TournamentSelect(rng, ga_params.selection_size),
        SelectionKeys.RANDOM_SELECT: lambda: RandomSelect(rng),
    }

    if not o_config.selection_types:
        logger.warning("No selection types enabled in the configuration. Selection will not occur.")
        return

    for variant_name in o_config.selection_types:
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
        for _ in range(num_parents):
            tournament_contenders = self.rng.sample(population, k=min(self.tournament_size, len(population)))
            winner = tournament_contenders[0]
            for contender in tournament_contenders[1:]:
                if contender.rank < winner.rank:
                    winner = contender
                if contender.rank == winner.rank and self.rng.choice([True, False]):
                    winner = contender
            yield winner


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals randomly from the population."""
        yield from self.rng.sample(population, k=num_parents)
