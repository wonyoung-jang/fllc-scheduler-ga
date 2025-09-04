"""Fitness evaluator for the FLL Scheduler GA."""

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from logging import getLogger
from math import sqrt
from typing import Any

from ..config.benchmark import FitnessBenchmark
from ..config.config import TournamentConfig
from ..config.constants import FitnessObjective, HardConstraint
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
    cache_info_success: dict[Any, Counter] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(tuple(FitnessObjective))
        self.cache_info_success = {k: Counter() for k in self.objectives}
        self.bt_cache = self.benchmark.timeslots
        self.tc_cache = self.benchmark.locations
        self.ov_cache = self.benchmark.opponents

    def check(self, schedule: Schedule) -> bool:
        """Check if the schedule meets hard constraints.

        Args:
            schedule (Schedule): The schedule to check.

        Returns:
            bool: True if the schedule meets all hard constraints, False otherwise.

        """
        # Check if the schedule is empty
        if not schedule:
            logger.debug("%s: %s", HardConstraint.SCHEDULE_EXISTENCE, "Schedule is empty")
            return False

        # Check if all events are scheduled
        if len(schedule) < self.config.total_slots:
            logger.debug("%s: %s", HardConstraint.ALL_EVENTS_SCHEDULED, "Not all events are scheduled")
            return False

        # Check team round requirements
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

        """
        if not self.check(schedule):
            schedule.fitness = None
            return

        scores = self.aggregate_team_fitnesses(schedule.all_teams())
        mean_s = self.get_mean_scores(scores)
        vari_s = self.get_variation_scores(scores, mean_s)
        mw, vw = self.config.weights

        schedule.fitness = tuple((m * mw) + (v * vw) for m, v in zip(mean_s, vari_s, strict=True))

    def aggregate_team_fitnesses(self, all_teams: list[Team]) -> tuple[list[float], ...]:
        """Aggregate fitness scores for all teams in the schedule."""
        scores: dict[FitnessObjective, list[float]] = defaultdict(list)
        cache_map = {
            FitnessObjective.BREAK_TIME: (self.hit_bt_cache, self.bt_cache),
            FitnessObjective.LOCATION_CONSISTENCY: (self.hit_tc_cache, self.tc_cache),
            FitnessObjective.OPPONENT_VARIETY: (self.hit_ov_cache, self.ov_cache),
        }

        for team in all_teams:
            team_fitness = []
            for obj, key in zip(self.objectives, team.get_fitness_keys(), strict=True):
                hit_cache, main_cache = cache_map[obj]
                if (val := hit_cache.get(key)) is None:
                    val = hit_cache.setdefault(key, main_cache.pop(key))
                    self.cache_info_success[obj]["miss"] += 1
                else:
                    self.cache_info_success[obj]["hit"] += 1

                scores[obj].append(val)
                team_fitness.append(val)

            team.fitness = tuple(team_fitness)

        return tuple(scores[obj] for obj in self.objectives)

    def get_mean_scores(self, scores: tuple[list[float], ...]) -> tuple[float, ...]:
        """Calculate the mean scores for each objective."""
        return tuple(sum(s) / self.config.num_teams for s in scores)

    def get_variation_scores(self, scores: tuple[list[float], ...], means: tuple[float, ...]) -> Iterator[float]:
        """Calculate the coefficient of variation for each objective."""
        _std_devs = (
            sqrt(sum((x - mean) ** 2 for x in lst) / self.config.num_teams)
            for lst, mean in zip(scores, means, strict=True)
        )
        _coeff_of_vars = (std_dev / mean if mean else 0 for std_dev, mean in zip(_std_devs, means, strict=True))
        yield from (1 / (1 + coeff) if coeff else 1 for coeff in _coeff_of_vars)

    def cache_info(self) -> dict[FitnessObjective, Counter[str, int]]:
        """Get cache information for fitness evaluations."""
        return self.cache_info_success
