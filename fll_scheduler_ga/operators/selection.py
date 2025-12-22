"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config.constants import SelectionOp


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    rng: np.random.Generator

    @abstractmethod
    def select(self, n: int, k: int) -> np.ndarray:
        """Select individuals from the population to form the next generation.

        Args:
            n (int): The population size to select from.
            k (int): The number to select.

        Returns:
            np.ndarray: The indices of the selected individuals.

        """


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def __str__(self) -> str:
        """Return a string representation of the selection operator."""
        return SelectionOp.RANDOM_SELECT

    def select(self, n: int, k: int = 2) -> np.ndarray:
        """Select individuals from the population to form the next generation."""
        if k == 2:
            # Two random indices
            i1 = self.rng.integers(0, n)
            i2 = self.rng.integers(0, n)

            # Ensure distinct
            while i1 == i2:
                i2 = self.rng.integers(0, n)

            return np.array([i1, i2], dtype=int)

        choices = np.arange(n)
        self.rng.shuffle(choices)
        return choices[:k]
