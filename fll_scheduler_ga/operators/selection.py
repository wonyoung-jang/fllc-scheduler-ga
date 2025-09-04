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
    from collections.abc import Iterator
    from random import Random

    from ..data_model.schedule import Population, Schedule


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    rng: Random

    @abstractmethod
    def select(self, population: Population, k: int = 2) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation.

        Args:
            population (Population): The current population of schedules.
            k (int): The number to select.

        Yields:
            Schedule: The selected parent schedules.

        """


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def __str__(self) -> str:
        """Return a string representation of the selection operator."""
        return SelectionOp.RANDOM_SELECT

    def select(self, population: Population, k: int = 2) -> Iterator[Schedule]:
        """Select individuals from the population to form the next generation."""
        yield from self.rng.sample(population, k=k)
