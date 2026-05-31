"""Stagnation handler for GA."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FitnessHistory:
    """Abstract base class for fitness history tracking."""

    generation: int
    current: np.ndarray
    history: np.ndarray

    def copy(self) -> FitnessHistory:
        """Create a copy of the fitness history."""
        return FitnessHistory(generation=self.generation, current=self.current.copy(), history=self.history.copy())

    def get_last_gen_fitness(self) -> np.ndarray:
        """Get the fitness of the last generation."""
        return (
            self.history[self.generation - 1] if self.generation > 0 else np.zeros(self.history.shape[1], dtype=float)
        )

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitnesses."""
        self.history[self.generation] = self.current
        self.generation += 1


@dataclass(slots=True)
class StagnationHandler:
    """Class for handling stagnation in the genetic algorithm."""

    enabled: bool
    threshold: int
    proportion: float
    cooldown: int
    _last_stagnant_gen: int = 0

    def is_stagnant(self, curr: int, history: np.ndarray) -> bool:
        """Check if the GA has stagnated based on fitness history."""
        if not self.enabled or curr < self.threshold:
            return False
        # Get recent history
        recents = history[curr - self.threshold : curr]
        # Checks if any of the recent fitnesses exactly the same as the first in this range
        equal_mask = recents[0] == recents[1:]
        # Count how many are equal in all objectives
        equal_sum = equal_mask.sum(axis=1) > 0
        equal_count = equal_sum.sum()
        # Determine stagnation
        if equal_count > self.threshold * self.proportion and curr - self._last_stagnant_gen >= self.cooldown:
            self._last_stagnant_gen = curr
            return True
        return False


@dataclass(slots=True)
class OperatorStats:
    """Class for collecting statistics on genetic operators."""

    offspring: Counter
    crossover: dict[str, Counter]
    mutation: dict[str, Counter]

    def count_offspring(self, op_status: str) -> None:
        """Record the use of an offspring operator (adding to population)."""
        self.offspring[op_status] += 1

    def count_crossover(self, op_status: str, op_name: str) -> None:
        """Record the use of a crossover operator."""
        self.crossover[op_status][op_name] += 1

    def count_mutation(self, op_status: str, op_name: str) -> None:
        """Record the use of a mutation operator."""
        self.mutation[op_status][op_name] += 1

    def get_offspring_stats(self) -> tuple[int, int, str]:
        """Get the offspring statistics."""
        s_sum = self.offspring.get("success", 0)
        t_sum = self.offspring.get("total", 0)
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate

    def _get_operator_stats(self, operators: dict[str, Counter]) -> tuple[int, int, str]:
        """Get the statistics for a specific set of operators."""
        s_sum = sum(operators.get("success", Counter()).values())
        t_sum = sum(operators.get("total", Counter()).values())
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate

    def get_crossover_stats(self) -> tuple[int, int, str]:
        """Get the crossover statistics for a specific operator."""
        return self._get_operator_stats(self.crossover)

    def get_mutation_stats(self) -> tuple[int, int, str]:
        """Get the mutation statistics for a specific operator."""
        return self._get_operator_stats(self.mutation)
