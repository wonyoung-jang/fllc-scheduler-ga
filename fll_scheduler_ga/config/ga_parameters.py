"""Parameters for the genetic algorithm used in FLL Scheduler."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GaParameters:
    """Parameters for the genetic algorithm."""

    population_size: int
    generations: int
    offspring_size: int
    selection_size: int
    crossover_chance: float
    mutation_chance: float
    num_islands: int
    migration_interval: int
    migration_size: int

    def __post_init__(self) -> None:
        """Post-initialization to ensure valid parameters."""
        if self.generations <= 0:
            self.generations = 128
            logger.warning("Generations must be greater than 0, defaulting to 128.")

        if not (0.0 < self.crossover_chance <= 1.0):
            self.crossover_chance = 0.5
            logger.warning("Crossover chance must be between 0.0 and 1.0, defaulting to 0.5.")

        if 2 <= self.population_size <= self.selection_size:
            self.population_size = self.selection_size + 1
            logger.warning("Population size must be greater than selection size, defaulting to selection_size + 1.")

        if self.offspring_size < 0:
            self.offspring_size = 0
            logger.warning("Offspring size cannot be negative, defaulting to 0.")

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

        logger.debug("Initialized genetic algorithm parameters: %s", self)

    def __str__(self) -> str:
        """Representation of GA parameters."""
        return (
            f"GaParameters:\n"
            f"\tPopulation Size: {self.population_size}\n"
            f"\tGenerations: {self.generations}\n"
            f"\tOffspring Size: {self.offspring_size}\n"
            f"\tSelection Size: {self.selection_size}\n"
            f"\tCrossover Chance: {self.crossover_chance:.2f}\n"
            f"\tMutation Chance: {self.mutation_chance:.2f}\n"
            f"\tNumber of Islands: {self.num_islands}\n"
            f"\tMigration Interval: {self.migration_interval}\n"
            f"\tMigration Size: {self.migration_size}"
        )
