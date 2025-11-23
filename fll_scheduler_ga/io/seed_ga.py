"""Seed data I/O for genetic algorithm."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import DATA_MODEL_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import TournamentConfig
    from ..data_model.schedule import Schedule
    from ..fitness.fitness import FitnessEvaluator

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
    evaluator: FitnessEvaluator

    def load(self) -> GASeedData | None:
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

        pop = []
        if seed_ga_data.version != DATA_MODEL_VERSION:
            logger.warning(
                "Seed population data version mismatch: Expected (%d), found (%d). Dismissing old seed file...",
                DATA_MODEL_VERSION,
                seed_ga_data.version,
            )
        elif seed_ga_data.config != self.config:
            logger.warning("Seed population does not match current config. Using current...")
            seed_ga_data.config = self.config
        elif not seed_ga_data.population:
            logger.warning("Seed population is missing. Using current...")
        else:
            pop = seed_ga_data.population

        # Handle changes in fitness weights to not flush cache
        if pop is not None and seed_ga_data.config.weights != self.config.weights:
            logger.info(
                "Updating seed population fitnesses to match current weights. Old weights: %s, New weights: %s",
                seed_ga_data.config.weights,
                self.config.weights,
            )
            pop_arr = np.array([s.schedule for s in pop], dtype=int)
            schedule_fitness, team_fitnesses = self.evaluator.evaluate_population(pop_arr)
            for i, schedule in enumerate(pop):
                schedule.fitness = schedule_fitness[i]
                schedule.team_fitnesses = team_fitnesses[i]
            pop.sort(key=lambda s: -s.fitness.sum())

        seed_ga_data.population = pop
        return seed_ga_data


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
