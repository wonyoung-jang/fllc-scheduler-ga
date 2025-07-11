"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

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
class Elitism(Selection):
    """Elitism for NSGA-II.

    Selects the best individuals for the next generation by taking whole fronts until the population size is met.
    """

    def select(self, population: Population, population_size: int) -> Iterator[Schedule]:
        """Select the new generation based on non-dominated sorting and crowding distance."""
        population.sort(key=lambda p: (p.rank, -p.crowding, -sum(p.fitness)))
        yield from population[:population_size]


@dataclass(slots=True)
class RouletteWheel(Selection):
    """Roulette wheel selection for genetic algorithms.

    Selects individuals based on their fitness proportionate to the total fitness of the population.
    """

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using roulette wheel selection."""
        total_fitness = sum(sum(schedule.fitness) for schedule in population if schedule.fitness is not None)

        if total_fitness == 0:
            msg = "Total fitness is zero, cannot perform roulette wheel selection."
            raise ValueError(msg)

        picks = sorted(self.rng.uniform(0, total_fitness) for _ in range(num_parents))
        current = 0

        for pick in picks:
            for schedule in population:
                if schedule.fitness is not None:
                    current += sum(schedule.fitness)

                    if current >= pick:
                        yield schedule


@dataclass(slots=True)
class RankBased(Selection):
    """Rank selection for genetic algorithms.

    Selects individuals based on their rank in the population.
    Higher-ranked individuals have a higher chance of being selected.
    """

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals based on their rank."""
        population.sort(key=lambda p: (p.rank, -p.crowding, -sum(p.fitness)), reverse=True)
        picks = sorted(self.rng.uniform(0, 1) for _ in range(num_parents))
        sp = self.rng.uniform(1, 2)
        n = len(population)

        for pick in picks:
            current = 0
            for i, schedule in enumerate(population):
                current += (2 - sp) + (sp - 1) * (i - 1) / (n - 1)
                if current >= pick:
                    yield schedule
                    break


@dataclass(slots=True)
class StochasticUniversalSampling(Selection):
    """Stochastic universal sampling for genetic algorithms.

    Selects individuals based on their fitness proportionate to the total fitness of the population.
    Uses a single random point to select multiple individuals.
    """

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using stochastic universal sampling."""
        total_fitness = sum(sum(schedule.fitness) for schedule in population if schedule.fitness is not None)

        if total_fitness == 0:
            msg = "Total fitness is zero, cannot perform stochastic universal sampling."
            raise ValueError(msg)

        distance = total_fitness / num_parents
        start = self.rng.uniform(0, distance)
        current = start

        for schedule in population:
            if schedule.fitness is not None:
                current += sum(schedule.fitness)
                while current >= distance:
                    yield schedule
                    current -= distance


@dataclass(slots=True)
class TournamentSelect(Selection):
    """Tournament selection for multi-objective problems using NSGA-II principles.

    Selects the winner based on rank, then crowding distance.
    """

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals using NSGA-II tournament selection."""
        for _ in range(num_parents):
            tournament = self.rng.sample(population, self.ga_parameters.selection_size)
            yield min(tournament, key=lambda p: (p.rank, -p.crowding, -sum(p.fitness)))


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def select(self, population: Population, num_parents: int) -> Iterator[Schedule]:
        """Select individuals randomly from the population."""
        yield from self.rng.sample(population, num_parents)
