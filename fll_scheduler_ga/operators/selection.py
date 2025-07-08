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
            participants = self.rng.sample(population, k=self.ga_parameters.selection_size)
            yield min(participants, key=lambda p: (p.rank, -p.crowding))


@dataclass(slots=True)
class ElitismSelectionNSGA2(Selection):
    """Elitism for NSGA-II.

    Selects the best individuals for the next generation by taking whole fronts until the population size is met.
    """

    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select the new generation based on non-dominated sorting and crowding distance."""
        population.sort(key=lambda p: p.rank)

        new_pop = []
        last_rank_start_index = 0
        for i, p in enumerate(population):
            if p.rank != population[last_rank_start_index].rank:
                current_front = population[last_rank_start_index:i]
                if len(new_pop) + len(current_front) > population_size:
                    current_front.sort(key=lambda ind: ind.crowding, reverse=True)
                    remaining_size = population_size - len(new_pop)
                    new_pop.extend(current_front[:remaining_size])
                    yield from new_pop
                    return
                new_pop.extend(current_front)
                last_rank_start_index = i

        last_front = population[last_rank_start_index:]
        if len(new_pop) + len(last_front) > population_size:
            last_front.sort(key=lambda ind: ind.crowding, reverse=True)
            remaining_size = population_size - len(new_pop)
            new_pop.extend(last_front[:remaining_size])
        else:
            new_pop.extend(last_front)

        yield from new_pop
