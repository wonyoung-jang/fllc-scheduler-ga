"""Population data module."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SchedulePopulation:
    """Population of schedules in the genetic algorithm."""

    ranks: np.ndarray
    schedules: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        """Return the number of schedules in the population."""
        return self.schedules.shape[0] if self.schedules is not None else 0

    def add(self, schedule: np.ndarray) -> None:
        """Add a new schedule to the population."""
        if self.schedules.size == 0:
            self.schedules = np.array([schedule], dtype=int)
        else:
            self.schedules = np.stack((*self.schedules, schedule), axis=0)
