"""Fitness evaluator for the FLL Scheduler GA."""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt
from typing import Any

from ..config.benchmark import FitnessBenchmark
from ..config.config import TournamentConfig
from .schedule import Schedule

logger = logging.getLogger(__name__)


class HardConstraints(StrEnum):
    """Enumeration of hard constraints for the FLL Scheduler GA."""

    ALL_EVENTS_SCHEDULED = "AllEventsScheduled"
    SCHEDULE_EXISTENCE = "ScheduleExistence"


class FitnessObjective(StrEnum):
    """Enumeration of fitness objectives for the FLL Scheduler GA."""

    BREAK_TIME = "BreakTime"
    TABLE_CONSISTENCY = "TableConsistency"
    OPPONENT_VARIETY = "OpponentVariety"


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    benchmark: FitnessBenchmark
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)
    hit_bt_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    hit_tc_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    hit_ov_cache: dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    _bt_cache: dict[Any, float] = field(default=None, init=False, repr=False)
    _tc_cache: dict[Any, float] = field(default=None, init=False, repr=False)
    _ov_cache: dict[Any, float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(
            [
                FitnessObjective.BREAK_TIME,
                FitnessObjective.TABLE_CONSISTENCY,
                FitnessObjective.OPPONENT_VARIETY,
            ]
        )
        self._bt_cache = self.benchmark.timeslots
        self._tc_cache = self.benchmark.table
        self._ov_cache = self.benchmark.opponents

    def check(self, schedule: Schedule) -> bool:
        """Check if the schedule meets hard constraints.

        Args:
            schedule (Schedule): The schedule to check.

        Returns:
            bool: True if the schedule meets all hard constraints, False otherwise.

        """
        # Check if the schedule is empty
        if not schedule:
            logger.debug("%s: %s", HardConstraints.SCHEDULE_EXISTENCE, "Schedule is empty")
            return False

        # Check if all events are scheduled
        if len(schedule) < self.config.total_slots:
            logger.debug("%s: %s", HardConstraints.ALL_EVENTS_SCHEDULED, "Not all events are scheduled")
            return False

        return True

    def evaluate(self, schedule: Schedule) -> tuple[float, ...] | None:
        """Evaluate the fitness of a schedule.

        Args:
            schedule (Schedule): The schedule to evaluate.

        Returns:
            tuple[float, ...] | None:
                A tuple of fitness scores for each objective or None if the schedule does not meet hard constraints.

            Objectives:
                - (bt) Break Time: Break time consistency across all teams.
                - (tc) Table Consistency: Table consistency across all teams.
                - (ov) Opponent Variety: Opponent variety across all teams.
            Metrics:
                - Mean: Average score across all teams for each objective.
                - Coefficient of Variation: Variation relative to the mean for each objective.
                - Range: Difference between the maximum and minimum scores for each objective.

        """
        if not self.check(schedule) or not (all_teams := schedule.all_teams()):
            return None

        bt_list = []
        tc_list = []
        ov_list = []

        for team in all_teams:
            bt_key = team.break_time_key()
            tc_key = team.table_consistency_key()
            ov_key = team.opponent_variety_key()

            if (t_bt := self.hit_bt_cache.get(bt_key)) is None:
                t_bt = self.hit_bt_cache.setdefault(bt_key, self._bt_cache.pop(bt_key))

            if (t_tc := self.hit_tc_cache.get(tc_key)) is None:
                t_tc = self.hit_tc_cache.setdefault(tc_key, self._tc_cache.pop(tc_key))

            if (t_ov := self.hit_ov_cache.get(ov_key)) is None:
                t_ov = self.hit_ov_cache.setdefault(ov_key, self._ov_cache.pop(ov_key))

            team.fitness = (t_bt, t_tc, t_ov)

            bt_list.append(t_bt)
            tc_list.append(t_tc)
            ov_list.append(t_ov)

        score_lists = (bt_list, tc_list, ov_list)

        # Metric 1: Averages of scores across all teams
        # The higher, the better
        totals = (sum(lst) for lst in score_lists)
        means = tuple(total / self.config.num_teams for total in totals)

        # Metric 2: Coefficient of Variation (CV) for each score, how much variation relative to mean
        # The lower, the better
        _sum_sq_diffs = (sum((x - mean) ** 2 for x in lst) for lst, mean in zip(score_lists, means, strict=True))
        _std_devs = (sqrt(sum_sq_diff / self.config.num_teams) for sum_sq_diff in _sum_sq_diffs)
        _coeff_of_vars = (std_dev / mean if mean else 0 for std_dev, mean in zip(_std_devs, means, strict=True))
        ratios = (1 / (1 + coeff) if coeff else 1 for coeff in _coeff_of_vars)

        # Metric 3: Range of scores for each objective (max - min)
        # The lower, the better
        _ranges = (max(lst) - min(lst) if lst else 1 for lst in score_lists)
        range_coeffs = (1 / (1 + range_val) if range_val else 1 for range_val in _ranges)

        return tuple(
            mean * ratio * range_coeff for mean, ratio, range_coeff in zip(means, ratios, range_coeffs, strict=True)
        )
