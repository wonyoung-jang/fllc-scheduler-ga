"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from math import sqrt
from typing import TYPE_CHECKING, Any

from ..config.constants import FitnessObjective, HardConstraint

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.benchmark import FitnessBenchmark
    from ..config.config import TournamentConfig
    from ..data_model.schedule import Schedule
    from ..data_model.team import Team

logger = getLogger(__name__)


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    benchmark: FitnessBenchmark
    bt_cache: dict[Any, float] = field(default=None, init=False, repr=False)
    tc_cache: dict[Any, float] = field(default=None, init=False, repr=False)
    ov_cache: dict[Any, float] = field(default=None, init=False, repr=False)
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)
    hit_bt_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    hit_tc_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    hit_ov_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    cache_info: dict[Any, Counter] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(list(FitnessObjective))
        self.cache_info = {k: Counter() for k in self.objectives}
        self.bt_cache = self.benchmark.timeslots
        self.tc_cache = self.benchmark.locations
        self.ov_cache = self.benchmark.opponents

    def check(self, schedule: Schedule) -> bool:
        """Check if the schedule meets hard constraints."""
        if not schedule:
            logger.debug("%s: %s", HardConstraint.SCHEDULE_EXISTENCE, "Schedule is empty")
            return False

        if len(schedule) < self.config.total_slots:
            logger.debug("%s: %s", HardConstraint.ALL_EVENTS_SCHEDULED, "Not all events are scheduled")
            return False

        if any(team.rounds_needed() for team in schedule.all_teams()):
            logger.debug("%s: %s", HardConstraint.TEAM_REQUIREMENTS_MET, "Some teams have unmet round requirements")
            return False

        return True

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
        if not self.check(schedule):
            schedule.fitness = None
            return

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
        cache_map = {
            FitnessObjective.BREAK_TIME: (self.hit_bt_cache, self.bt_cache),
            FitnessObjective.LOCATION_CONSISTENCY: (self.hit_tc_cache, self.tc_cache),
            FitnessObjective.OPPONENT_VARIETY: (self.hit_ov_cache, self.ov_cache),
        }
        cache_info = self.cache_info
        for team in all_teams:
            keys = team.get_fitness_keys()
            vals = []
            for obj, key in zip(objectives, keys, strict=True):
                hit_cache, main_cache = cache_map[obj]
                if (val := hit_cache.get(key)) is None:
                    val = hit_cache.setdefault(key, main_cache.pop(key))
                    cache_info[obj]["miss"] += 1
                else:
                    cache_info[obj]["hit"] += 1
                scores[obj].append(val)
                vals.append(val)
            team.fitness = tuple(vals)
        return tuple(scores[obj] for obj in scores)

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

    def get_cache_info(self) -> dict[FitnessObjective, Counter[str, int]]:
        """Get cache information for fitness evaluations."""
        return self.cache_info
