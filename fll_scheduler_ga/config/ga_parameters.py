"""Parameters for the genetic algorithm used in FLL Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any

logger = getLogger(__name__)


@dataclass(slots=True)
class GaParameters:
    """Parameters for the genetic algorithm."""

    population_size: int
    generations: int
    offspring_size: int
    crossover_chance: float
    mutation_chance: float
    num_islands: int
    migration_interval: int
    migration_size: int

    def __post_init__(self) -> None:
        """Post-initialization to ensure valid parameters."""
        self._validate()
        logger.debug("Initialized genetic algorithm parameters: %s", self)

    def __str__(self) -> str:
        """Representation of GA parameters."""
        return (
            f"\n\tGaParameters:"
            f"\n\t  population_size    : {self.population_size}"
            f"\n\t  generations        : {self.generations}"
            f"\n\t  offspring_size     : {self.offspring_size}"
            f"\n\t  crossover_chance   : {self.crossover_chance:.2f}"
            f"\n\t  mutation_chance    : {self.mutation_chance:.2f}"
            f"\n\t  num_islands        : {self.num_islands}"
            f"\n\t  migration_interval : {self.migration_interval}"
            f"\n\t  migration_size     : {self.migration_size}"
        )

    @classmethod
    def build(cls, params: dict[str, Any]) -> GaParameters:
        """Build GaParameters from a dictionary of parameters."""
        return cls(**params)

    def _validate(self) -> None:
        """Validate the parameters."""
        if self.generations <= 0:
            self.generations = 128
            logger.warning("Generations must be greater than 0, defaulting to %d.", self.generations)

        if not (0.0 < self.crossover_chance <= 1.0):
            self.crossover_chance = 0.6
            logger.warning("Crossover chance must be between 0.0 and 1.0, defaulting to %f.", self.crossover_chance)

        if self.population_size <= 1:
            self.population_size = 2
            logger.warning("Population size must be greater than 1, defaulting to %d.", self.population_size)

        if self.offspring_size < 0:
            self.offspring_size = 0
            logger.warning("Offspring size cannot be negative, defaulting to %d.", self.offspring_size)

        if not (0.0 <= self.mutation_chance <= 1.0):
            self.mutation_chance = 0.2
            logger.warning("Mutation chance must be between 0.0 and 1.0, defaulting to %f.", self.mutation_chance)

        if self.num_islands < 1:
            self.num_islands = 1
            logger.warning("Number of islands must be at least 1, defaulting to %d.", self.num_islands)

        if self.migration_interval < 0:
            self.migration_interval = 0
            logger.warning("Migration interval must not be negative, defaulting to %d.", self.migration_interval)

        if self.migration_size < 0:
            self.migration_size = 0
            logger.warning("Migration size cannot be negative, defaulting to %d.", self.migration_size)

        if self.num_islands > 1 and self.migration_size >= self.population_size:
            self.migration_size = max(1, self.population_size // 5)
            logger.warning("Migration size is >= population size, defaulting to max(1, 20%%): %i", self.migration_size)
