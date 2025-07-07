"""Observers for the FLL Scheduler GA."""

from dataclasses import dataclass
from logging import Logger

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

    def on_finish(self) -> None:
        """Log the completion of the genetic algorithm run."""
        self.logger.debug("Genetic algorithm run completed.")

    def on_mutation(self, mutation_name: str) -> None:
        """Log mutation details."""
        self.logger.debug("Mutation applied: %s", mutation_name)

    def on_crossover(self, crossover_name: str, *, successful: bool) -> None:
        """Log crossover details."""
        if not successful:
            self.logger.debug("Crossover failed: %s", crossover_name)
        else:
            self.logger.debug("Crossover applied: %s", crossover_name)
