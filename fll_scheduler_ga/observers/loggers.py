"""Observers for the FLL Scheduler GA."""

from dataclasses import dataclass
from logging import getLogger

from ..data_model.schedule import Population
from .base_observer import GaObserver

logger = getLogger(__name__)


@dataclass(slots=True)
class LoggingObserver(GaObserver):
    """Observer that logs generation and best fitness information."""

    def on_start(self, num_generations: int) -> None:
        """Log the start of the genetic algorithm run."""
        logger.debug("Starting genetic algorithm run for %d generations.", num_generations)

    def on_generation_end(
        self,
        generation: int,
        num_generations: int,
        population_size: int,
        best_fitness: tuple[float, ...],
        front_size: int,
    ) -> None:
        """Log the end of a generation with population size and best fitness."""
        logger.debug(
            "Population: %d/%d | Generation %d/%d | Best Fitness %s",
            front_size,
            population_size,
            generation,
            num_generations,
            best_fitness,
        )

    def on_finish(self, pop: Population, front: Population) -> None:
        """Log the completion of the genetic algorithm run."""
        logger.debug("Genetic algorithm run completed.")
        if not pop:
            logger.warning("No valid schedule was found after all generations.")
            return
        logger.debug("Final pareto front size: %d/%d (%.2f%%)", len(front), len(pop), len(front) / len(pop) * 100)
