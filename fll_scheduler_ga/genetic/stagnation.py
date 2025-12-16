"""Stagnation handler for GA."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config.schemas import StagnationModel
    from .ga_generation import GaGeneration

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessHistory:
    """Abstract base class for fitness history tracking."""

    generation: GaGeneration
    current: np.ndarray
    history: np.ndarray

    def copy(self) -> FitnessHistory:
        """Create a copy of the fitness history."""
        return FitnessHistory(
            generation=self.generation,
            current=self.current.copy(),
            history=self.history.copy(),
        )

    def get_last_gen_fitness(self) -> np.ndarray:
        """Get the fitness of the last generation."""
        return (
            self.history[self.generation.curr - 1]
            if self.generation.curr > 0
            else np.zeros(self.history.shape[1], dtype=float)
        )

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitnesses."""
        self.history[self.generation.curr] = self.current


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

    def get_crossover_stats(self) -> tuple[int, int, str]:
        """Get the crossover statistics for a specific operator."""
        s = self.crossover.get("success", Counter())
        t = self.crossover.get("total", Counter())
        s_sum = sum(s.values())
        t_sum = sum(t.values())
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate

    def get_mutation_stats(self) -> tuple[int, int, str]:
        """Get the mutation statistics for a specific operator."""
        s = self.mutation.get("success", Counter())
        t = self.mutation.get("total", Counter())
        s_sum = sum(s.values())
        t_sum = sum(t.values())
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate


@dataclass(slots=True)
class StagnationHandler:
    """Class for handling stagnation in the genetic algorithm."""

    rng: np.random.Generator
    generation: GaGeneration
    fitness_history: FitnessHistory
    model: StagnationModel

    _last_stagnant_gen: int = 0

    def is_stagnant(self) -> bool:
        """Check if the GA has stagnated based on fitness history."""
        if not self.model.enable:
            return False

        curr = self.generation.curr
        if curr < self.model.threshold:
            return False

        # Get recent history
        recents = self.fitness_history.history[curr - self.model.threshold : curr]

        # Checks if any of the recent fitnesses exactly the same as the first in this range
        equal_mask = recents[0] == recents[1:]

        # Count how many are equal in all objectives
        equal_sum = equal_mask.sum(axis=1) > 0
        equal_count = equal_sum.sum()

        # Determine stagnation
        if equal_count > self.model.threshold * self.model.proportion:
            if curr - self._last_stagnant_gen < self.model.cooldown:
                return False
            self._last_stagnant_gen = curr
            return True
        return False
