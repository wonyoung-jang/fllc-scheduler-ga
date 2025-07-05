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

    def __post_init__(self) -> None:
        """Post-initialization to ensure valid parameters."""
        if self.generations <= 0:
            self.generations = 10
            logger.warning("Generations must be greater than 0, defaulting to 10.")

        if not (0.0 <= self.crossover_chance <= 1.0):
            self.crossover_chance = 0.5
            logger.warning("Crossover chance must be between 0.0 and 1.0, defaulting to 0.5.")

        if self.population_size < 2:
            self.population_size = 2
            logger.warning("Population size must be at least 2, defaulting to 2.")

        if self.elite_size < 0:
            self.elite_size = 0
            logger.warning("Elite size cannot be negative, defaulting to 0.")

        if self.elite_size >= self.population_size:
            self.elite_size = self.population_size // 10
            logger.warning("Elite size is greater than or equal to population size, defaulting to 10%%")

        if self.selection_size < 2:
            self.selection_size = 2
            logger.warning("Selection size must be at least 2, defaulting to 2.")

        if self.selection_size > self.population_size:
            self.selection_size = self.population_size // 5
            logger.warning("Selection size is greater than population size, defaulting to 20%%")
