"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..genetic.schedule import Population, Schedule


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
