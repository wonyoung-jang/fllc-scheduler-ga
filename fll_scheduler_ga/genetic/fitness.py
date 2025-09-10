"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from math import sqrt
from typing import TYPE_CHECKING, Any

from ..config.constants import FitnessObjective

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.benchmark import FitnessBenchmark
    from ..config.config import TournamentConfig
    from ..data_model.schedule import Schedule
    from ..data_model.team import Team

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
        """Evaluate the fitness of a schedule.

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
        scores = self.aggregate_team_fitnesses(schedule.all_teams())
        mean_s = self.get_mean_scores(scores)
        vari_s = self.get_variation_scores(scores, mean_s)
        rnge_s = self.get_range_scores(scores)
        mw, vw, rw = self.config.weights
        schedule.fitness = tuple(
            (m * mw) + (v * vw) + (r * rw)
            for m, v, r in zip(
                mean_s,
                vari_s,
                rnge_s,
                strict=True,
            )
        )

    def aggregate_team_fitnesses(self, all_teams: list[Team]) -> tuple[list[float], ...]:
        """Aggregate fitness scores for all teams in the schedule."""
        objectives = self.objectives
        scores: dict[FitnessObjective, list[float]] = {obj: [] for obj in objectives}
        cache_map = self.cache_map
        for team in all_teams:
            keys = team.get_fitness_keys()
            vals = []
            for obj, key in zip(objectives, keys, strict=True):
                cache = cache_map[obj]
                val = cache.get(key)
                if val is None:
                    logger.error("Fitness evaluation failed to retrieve value from cache for %s with key %s", obj, key)
                    val = 0.0
                scores[obj].append(val)
                vals.append(val)
            team.fitness = tuple(vals)
        return tuple(scores.values())

    def get_mean_scores(self, scores: tuple[list[float], ...]) -> tuple[float, ...]:
        """Calculate the mean scores for each objective."""
        return tuple(sum(s) / len(s) for s in scores)

    def get_variation_scores(self, scores: tuple[list[float], ...], means: tuple[float, ...]) -> Iterator[float]:
        """Calculate the coefficient of variation for each objective."""
        for lst, mean in zip(scores, means, strict=True):
            n = len(lst)
            if n == 0 or mean == 0:
                yield 1
                continue
            ss = sum((x - mean) ** 2 for x in lst)
            std_dev = sqrt(ss / n)
            coeff = std_dev / mean
            yield 1 / (1 + coeff) if coeff else 1

    def get_range_scores(self, scores: tuple[list[float], ...]) -> Iterator[float]:
        """Calculate the range of scores for each objective."""
        yield from (1 / (1 + max(lst) - min(lst)) if lst else 1 for lst in scores)

    def get_cache_info(self) -> dict[FitnessObjective, str]:
        """Get cache information for fitness evaluations."""
        return {obj: f"Hits: {cache.hits}, Misses: {cache.misses}" for obj, cache in self.cache_map.items()}
