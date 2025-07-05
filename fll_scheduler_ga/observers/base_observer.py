"""Observers for the FLL Scheduler GA."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GaObserver(ABC):
    """Abstract base class for observers in the FLL Scheduler GA."""

    @abstractmethod
    def on_start(self, num_generations: int) -> None:
        """Call at the start of the genetic algorithm run to initialize observers."""

    @abstractmethod
    def on_generation_end(
        self, generation: int, num_generations: int, population_size: int, best_fitness: float
    ) -> None:
        """Call at the end of each generation to report status."""

    @abstractmethod
    def on_finish(self) -> None:
        """Call when the genetic algorithm run is finished."""

    @abstractmethod
    def on_mutation(self, mutation_name: str) -> None:
        """Call when a mutation is applied."""

    @abstractmethod
    def on_crossover(self, crossover_name: str) -> None:
        """Call when a crossover is applied."""
