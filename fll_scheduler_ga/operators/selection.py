"""Selection operators for genetic algorithms in FLL scheduling."""

from abc import ABC
from collections import defaultdict
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
            participants = self.rng.sample(population, k=self.ga_parameters.selection_size)
            yield min(participants, key=lambda p: (p.rank, -p.crowding_distance))


@dataclass(slots=True)
class ElitismSelectionNSGA2(Selection):
    """Elitism for NSGA-II.

    Selects the best individuals for the next generation by taking whole fronts until the population size is met.
    """

    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select the new generation based on non-dominated sorting and crowding distance."""
        fronts = defaultdict(list)
        for p in population:
            fronts[p.rank].append(p)

        new_pop = []
        for i in sorted(fronts.keys()):
            front: Population = fronts[i]
            if len(new_pop) + len(front) <= population_size:
                new_pop.extend(front)
            else:
                front.sort(key=lambda p: p.crowding_distance, reverse=True)
                new_pop.extend(front[: population_size - len(new_pop)])
                break
        yield from new_pop
