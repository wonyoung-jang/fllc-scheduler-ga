"""Observers for the FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from rich.progress import Progress

    from ..data_model.schedule import Schedule


logger = getLogger(__name__)


@dataclass(slots=True)
class GaObserver(ABC):
    """Abstract base class for observers in the FLL Scheduler GA."""

    @abstractmethod
    def on_start(self, num_generations: int) -> None:
        """Call at the start of the genetic algorithm run to initialize observers."""

    @abstractmethod
    def on_generation_end(
        self,
        generation: int,
        num_generations: int,
        best_fitness: np.ndarray,
        pop_size: int,
    ) -> None:
        """Call at the end of each generation to report status."""

    @abstractmethod
    def on_finish(self, pop: list[Schedule], front: list[Schedule]) -> None:
        """Call when the genetic algorithm run is finished."""


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
        best_fitness: np.ndarray,
        pop_size: int,
    ) -> None:
        """Log the end of a generation with population size and best fitness."""
        fitness_str = "N/A"
        if best_fitness.any():
            fitness_str = ", ".join([f"{s:.2f}" for s in best_fitness])
            fitness_str += f" | Σ={sum(best_fitness):.2f} ({sum(best_fitness) / len(best_fitness):.1%})"
        logger.debug("Fitness %s | Generation %d/%d", fitness_str, generation, num_generations)

    def on_finish(self, pop: list[Schedule], front: list[Schedule]) -> None:
        """Log the completion of the genetic algorithm run."""
        logger.debug("Genetic algorithm run completed.")
        if not pop:
            logger.warning("No valid schedule was found after all generations.")
            return
        len_front = len(front)
        len_pop = len(pop)
        front_portion = len_front / len_pop * 100 if len_pop > 0 else 0.0
        logger.debug("Final pareto front size: %d/%d (%.2f%%)", len_front, len_pop, front_portion)


@dataclass(slots=True)
class RichObserver(GaObserver):
    """Connects GA progress to a Rich Progress Task."""

    progress: Progress
    task_id: int

    def on_start(self, num_generations: int) -> None:
        """Initialize progress task."""
        self.progress.update(self.task_id, total=num_generations)

    def on_generation_end(self, generation: int, num_generations: int, best_fitness: np.ndarray, pop_size: int) -> None:
        """Update progress task at generation end."""
        fit_val = sum(best_fitness) if best_fitness.any() else 0.0
        self.progress.update(
            self.task_id,
            completed=generation,
            description=f"[cyan]Gen {generation}[/cyan] | Fit: [green]{fit_val:.4f}[/green]",
        )

    def on_finish(self, pop: list, front: list) -> None:
        """Finalize progress task."""
