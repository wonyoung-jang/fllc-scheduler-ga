"""Selection operators for genetic algorithms in FLL scheduling.

The selection classes are sorted from highest to lowest selective pressure.

Higher selective pressure means better individuals have a higher chance
of being selected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.constants import SelectionOp

if TYPE_CHECKING:
    import numpy as np


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
        return self.rng.choice(n, size=k, replace=False)
