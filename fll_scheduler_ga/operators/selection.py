"""Selection operators for genetic algorithms in FLL scheduling."""

from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..genetic.ga_parameters import GaParameters
from ..genetic.schedule import Population, Schedule


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    ga_parameters: GaParameters
    rng: Random


@dataclass(slots=True)
class TournamentSelectionNSGA2(Selection):
    """Tournament selection for multi-objective problems using NSGA-II principles.

    Selects the winner based on rank, then crowding distance.
    """

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using NSGA-II tournament selection."""
        for _ in range(num_parents):
            tournament = self.rng.sample(population, self.ga_parameters.selection_size)
            yield min(tournament, key=lambda p: (p.rank, -p.crowding))


@dataclass(slots=True)
class ElitismSelectionNSGA2(Selection):
    """Elitism for NSGA-II.

    Selects the best individuals for the next generation by taking whole fronts until the population size is met.
    """

    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select the new generation based on non-dominated sorting and crowding distance."""
        yield from population[:population_size]
