"""Repository interface and implementations for benchmark data."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkSeedData:
    """Seed data object for fitness benchmarks."""

    version: int
    opponents: np.ndarray
    best_timeslot_score: float


@dataclass(slots=True)
class BenchmarkRepository(ABC):
    """Abstract interface for storing and retrieving benchmark data."""

    @abstractmethod
    def load(self) -> BenchmarkSeedData | None:
        """Load benchmark data."""

    @abstractmethod
    def save(self, data: BenchmarkSeedData) -> None:
        """Save benchmark data."""


@dataclass(slots=True)
class PickleBenchmarkRepository(BenchmarkRepository):
    """Concrete implementation using Pickle and local file system."""

    path: Path

    def load(self) -> BenchmarkSeedData | None:
        """Load benchmark data from a pickle file."""
        logger.debug("Loading fitness benchmarks from cache: %s", self.path)
        if not self.path.exists():
            return None

        try:
            with self.path.open("rb") as f:
                seed_data = pickle.load(f)
        except (OSError, EOFError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
            logger.debug("Failed to load fitness benchmarks from cache: %s", self.path)
            return None

        return seed_data

    def save(self, data: BenchmarkSeedData) -> None:
        """Save benchmark data to a pickle file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("wb") as f:
                pickle.dump(data, f)
            logger.info("Fitness benchmarks saved to cache: %s", self.path)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Failed to save fitness benchmarks to cache.")
