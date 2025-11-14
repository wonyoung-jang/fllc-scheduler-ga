"""Seed data I/O for fitness benchmarks."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


logger = getLogger(__name__)


@dataclass(slots=True)
class BenchmarkSeedData:
    """Seed data object for fitness benchmarks."""

    version: int
    opponents: np.ndarray
    best_timeslot_score: float


@dataclass(slots=True)
class BenchmarkLoad:
    """Loader for fitness benchmark data from cache files."""

    path: Path

    def load(self) -> BenchmarkSeedData | None:
        """Load benchmark data from a pickle file."""
        try:
            with self.path.open("rb") as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.exception("Failed to load fitness benchmarks from cache.")
            return None


@dataclass(slots=True)
class BenchmarkSave:
    """Saver for fitness benchmark data to cache files."""

    path: Path
    data: BenchmarkSeedData

    def save(self) -> None:
        """Save benchmark data to a pickle file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self.path.open("wb") as f:
                pickle.dump(self.data, f)
            logger.info("Fitness benchmarks saved to cache: %s", self.path)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Failed to save fitness benchmarks to cache.")
