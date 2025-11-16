"""Stagnation handler for GA."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessHistory:
    """Abstract base class for fitness history tracking."""

    curr_gen: int
    curr_fit: np.ndarray
    history: np.ndarray

    def get_last_gen_fitness(self) -> np.ndarray:
        """Get the fitness of the last generation."""
        return self.history[self.curr_gen - 1] if self.curr_gen > 0 else np.zeros(self.history.shape[1], dtype=float)

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitnesses."""
        self.history[self.curr_gen] = self.curr_fit
