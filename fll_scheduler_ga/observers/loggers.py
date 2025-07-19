"""Observers for the FLL Scheduler GA."""

from dataclasses import dataclass
from logging import Logger

from ..genetic.schedule import Population
from .base_observer import GaObserver


@dataclass(slots=True)
class LoggingObserver(GaObserver):
    """Observer that logs generation and best fitness information."""

    logger: Logger

    def on_start(self, num_generations: int) -> None:
        """Log the start of the genetic algorithm run."""

    def on_generation_end(
        self,
        generation: int,
        num_generations: int,
        population_size: int,
        best_fitness: tuple[float, ...],
        front_size: int,
    ) -> None:
        """Log the end of a generation with population size and best fitness."""
        self.logger.debug(
            "Population: %d | Generation %d/%d",
            population_size,
            generation,
            num_generations,
        )

    def on_finish(self, pop: Population, front: Population) -> None:
        """Log the completion of the genetic algorithm run."""
        self.logger.debug("Genetic algorithm run completed.")
        if not pop:
            self.logger.warning("No valid schedule was found after all generations.")
            return

        self.logger.info("Final pareto front size: %d", len(front))
