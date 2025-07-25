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

    num_islands: int = 10
    migration_interval: int = 10
    migration_size: int = 2

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

        if self.num_islands < 1:
            self.num_islands = 1
            logger.warning("Number of islands must be at least 1, defaulting to 1.")

        if self.migration_interval <= 0:
            self.migration_interval = 10
            logger.warning("Migration interval must be positive, defaulting to 10.")

        if self.migration_size < 0:
            self.migration_size = 0
            logger.warning("Migration size cannot be negative, defaulting to 0.")

        if self.num_islands > 1 and self.migration_size >= self.population_size:
            self.migration_size = max(1, self.population_size // 4)
            logger.warning("Migration size is >= population size, defaulting to 25%%.")

        logger.debug("Initialized GaParameters: %s", self)

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
            f"\tNumber of Islands: {self.num_islands}\n"
            f"\tMigration Interval: {self.migration_interval}\n"
            f"\tMigration Size: {self.migration_size}\n"
        )
