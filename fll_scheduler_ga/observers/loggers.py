"""Observers for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from .base_observer import GaObserver

if TYPE_CHECKING:
    from ..data_model.schedule import Population

logger = getLogger(__name__)


@dataclass(slots=True)
class LoggingObserver(GaObserver):
    """Observer that logs generation and best fitness information."""

    def on_start(self, num_generations: int) -> None:
        """Log the start of the genetic algorithm run."""
        logger.debug("Starting genetic algorithm run for %d generations.", num_generations)

    def on_generation_end(self, generation: int, num_generations: int, best_fitness: tuple[float, ...]) -> None:
        """Log the end of a generation with population size and best fitness."""
        fitness_str = "N/A"
        if best_fitness:
            fitness_str = ", ".join([f"{s:.2f}" for s in best_fitness])
            fitness_str += f" | Σ={sum(best_fitness):.2f} ({sum(best_fitness) / len(best_fitness):.1%})"
        logger.debug("Fitness %s | Generation %d/%d", fitness_str, generation, num_generations)

    def on_finish(self, pop: Population, front: Population) -> None:
        """Log the completion of the genetic algorithm run."""
        logger.debug("Genetic algorithm run completed.")
        if not pop:
            logger.warning("No valid schedule was found after all generations.")
            return
        len_front = len(front)
        len_pop = len(pop)
        front_portion = len_front / len_pop * 100 if len_pop > 0 else 0.0
        logger.debug("Final pareto front size: %d/%d (%.2f%%)", len_front, len_pop, front_portion)
