"""Parameters for the genetic algorithm used in FLL Scheduler."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GaParameters:
    """Parameters for the genetic algorithm."""

    population_size: int = 32
    generations: int = 128
    elite_size: int = 8
    selection_size: int = 16
    crossover_chance: float = 0.5
    mutation_chance: float = 0.05

    def __post_init__(self) -> None:
        """Post-initialization to ensure valid parameters."""
        if self.generations <= 0:
            self.generations = 10
            logger.warning("Generations must be greater than 0, defaulting to 10.")

        if not (0.0 < self.crossover_chance <= 1.0):
            self.crossover_chance = 0.5
            logger.warning("Crossover chance must be between 0.0 and 1.0, defaulting to 0.5.")

        if 2 <= self.population_size <= self.selection_size:
            self.population_size = self.selection_size + 1
            logger.warning("Population size must be greater than selection size, defaulting to selection_size + 1.")

        if self.elite_size < 0:
            self.elite_size = 0
            logger.warning("Elite size cannot be negative, defaulting to 0.")

        if self.elite_size >= self.population_size:
            self.elite_size = max(1, self.population_size // 10)
            logger.warning("Elite size is greater than or equal to population size, defaulting to 10%%")

        if self.selection_size < 2:
            self.selection_size = 2
            logger.warning("Selection size must be at least 2, defaulting to 2.")

        if self.selection_size >= self.population_size:
            self.selection_size = max(2, self.population_size // 5)
            logger.warning("Selection size is greater than or equal to population size, defaulting to 20%%")

        if not (0.0 <= self.mutation_chance <= 1.0):
            self.mutation_chance = 0.05
            logger.warning("Mutation chance must be between 0.0 and 1.0, defaulting to 0.05.")

    def __str__(self) -> str:
        """Representation of GA parameters."""
        return (
            f"GaParameters:\n"
            f"\tPopulation Size: {self.population_size}\n"
            f"\tGenerations: {self.generations}\n"
            f"\tElite Size: {self.elite_size}\n"
            f"\tSelection Size: {self.selection_size}\n"
            f"\tCrossover Chance: {self.crossover_chance:.2f}\n"
            f"\tMutation Chance: {self.mutation_chance:.2f}\n"
        )
