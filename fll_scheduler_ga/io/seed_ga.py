"""Seed data I/O for genetic algorithm."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import DATA_MODEL_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import TournamentConfig
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class GASeedData:
    """GA seed data object."""

    version: int
    config: TournamentConfig
    population: list[Schedule]


@dataclass(slots=True)
class GALoad:
    """Loader for GA instances from seed files."""

    seed_file: Path
    config: TournamentConfig

    def load(self) -> list[Schedule] | None:
        """Load and integrate a population from a seed file."""
        try:
            logger.debug("Loading seed population from: %s", self.seed_file)
            with self.seed_file.open("rb") as f:
                seed_ga_data: GASeedData = pickle.load(f)
        except (OSError, pickle.PicklingError):
            logger.exception("Could not load or parse seed file. Starting with a fresh population.")
            return None
        except EOFError:
            logger.debug("Pickle file is empty")
            return None

        output = None
        if seed_ga_data.version != DATA_MODEL_VERSION:
            logger.warning(
                "Seed population data version mismatch: Expected (%d), found (%d). Dismissing old seed file...",
                DATA_MODEL_VERSION,
                seed_ga_data.version,
            )
        elif seed_ga_data.config != self.config:
            logger.warning("Seed population does not match current config. Using current...")
        elif not seed_ga_data.population:
            logger.warning("Seed population is missing. Using current...")
        else:
            output = seed_ga_data.population

        return output


@dataclass(slots=True)
class GASave:
    """Saver for GA instances to seed files."""

    seed_file: Path
    data: GASeedData

    def save(self) -> None:
        """Save the final population to a file to be used as a seed for a future run."""
        try:
            seed_ga_data = self.data
            path = self.seed_file
            logger.debug("Saving final population of size %d to seed file: %s", len(seed_ga_data.population), path)
            with path.open("wb") as f:
                pickle.dump(seed_ga_data, f)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Error saving population to seed file: %s", path)
