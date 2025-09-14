"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.constants import FitnessObjective

if TYPE_CHECKING:
    from ..config.benchmark import FitnessBenchmark
    from ..config.config import TournamentConfig
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class HardConstraintChecker:
    """Validates hard constraints for a schedule."""

    config: TournamentConfig

    def check(self, schedule: Schedule) -> bool:
        """Check the hard constraints of a schedule."""
        if not schedule:
            return False

        if len(schedule) != self.config.total_slots:
            return False

        return not any(team.rounds_needed() for team in schedule.all_teams())


@dataclass(slots=True)
class TwoTierCache:
    """Two-tier cache for fitness evaluation."""

    hot_cache: dict[Any, float] = field(default_factory=dict)
    cold_cache: dict[Any, float] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: Any) -> float | None:
        """Get a value from the cache."""
        if key in self.hot_cache:
            self.hits += 1
            return self.hot_cache[key]

        if key in self.cold_cache:
            self.misses += 1
            value = self.cold_cache.pop(key)
            self.hot_cache[key] = value
            return value

        return None


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    benchmark: FitnessBenchmark
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)
    cache_map: dict[FitnessObjective, TwoTierCache] = None

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(list(FitnessObjective))
        self.cache_map = {
            FitnessObjective.BREAK_TIME: TwoTierCache(cold_cache=self.benchmark.timeslots),
            FitnessObjective.LOCATION_CONSISTENCY: TwoTierCache(cold_cache=self.benchmark.locations),
            FitnessObjective.OPPONENT_VARIETY: TwoTierCache(cold_cache=self.benchmark.opponents),
        }

    def evaluate(self, schedule: Schedule) -> None:
        """Evaluate the fitness of a schedule using vectorized NumPy operations.

        This is the primary performance-critical function in the GA. By creating one
        large NumPy array for all team scores, we can use fast, vectorized functions
        to calculate mean, variation, and range, avoiding slow Python loops.

        Args:
            schedule (Schedule): The schedule to evaluate.

        Objectives:
            - (bt) Break Time: Break time consistency across all teams.
            - (tc) Table Consistency: Table consistency across all teams.
            - (ov) Opponent Variety: Opponent variety across all teams.
        Metrics:
            - Mean: Average score across all teams for each objective.
            - Coefficient of Variation: Variation relative to the mean for each objective.
            - Range: Difference between the maximum and minimum scores for each objective.

        """
        all_teams = schedule.all_teams()
        num_teams = len(all_teams)
        if num_teams == 0:
            schedule.fitness = tuple([0.0] * len(self.objectives))
            return

        # Main "setup"
        team_scores_list = []
        for team in all_teams:
            vals = []
            keys = team.get_fitness_keys()
            for obj, key in zip(self.objectives, keys, strict=True):
                val = self.cache_map[obj].get(key)
                if val is None:
                    logger.error("Cache miss for %s with key %s", obj, key)
                    val = 0.0
                vals.append(val)
            team.fitness = tuple(vals)
            team_scores_list.append(vals)

        # Shape (num_teams, num_objectives)
        scores = np.array(team_scores_list, dtype=float)

        # Calculate mean scores for each objective (column-wise mean)
        mean_s = np.mean(scores, axis=0)

        # Calculate standard deviation for each objective
        std_devs = np.std(scores, axis=0)

        # Calculate Coefficient of Variation scores
        epsilon = 1e-12  # Avoid division by zero
        coeffs_of_variation = std_devs / (mean_s + epsilon)
        vari_s = 1 / (1 + coeffs_of_variation)

        # Calculate Range scores (vectorized)
        ranges = np.ptp(scores, axis=0)  # ptp = "peak to peak" (max - min) for each objective
        rnge_s = 1 / (1 + ranges)

        # Combine weighted scores into the final fitness tuple
        mw, vw, rw = self.config.weights
        final_scores = (mean_s * mw) + (vari_s * vw) + (rnge_s * rw)

        schedule.fitness = tuple(final_scores)

    def get_cache_info(self) -> dict[FitnessObjective, str]:
        """Get cache information for fitness evaluations."""
        return {obj: f"Hits: {cache.hits}, Misses: {cache.misses}" for obj, cache in self.cache_map.items()}
