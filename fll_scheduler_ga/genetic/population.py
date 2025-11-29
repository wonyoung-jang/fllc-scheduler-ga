"""Population data module."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SchedulePopulation:
    """Population of schedules in the genetic algorithm."""

    schedules: np.ndarray | None = None
    ranks: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the schedules array."""
        if self.ranks is None:
            self.ranks = np.empty((0,), dtype=int)

    def __len__(self) -> int:
        """Return the number of schedules in the population."""
        return self.schedules.shape[0] if self.schedules is not None else 0

    def add_schedule(self, schedule: np.ndarray) -> None:
        """Add a new schedule to the population."""
        if self.schedules is None:
            self.schedules = np.array([schedule], dtype=int)
        else:
            self.schedules = np.stack((*self.schedules, schedule), axis=0)
