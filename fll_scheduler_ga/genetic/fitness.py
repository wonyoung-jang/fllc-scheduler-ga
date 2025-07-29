"""Fitness evaluator for the FLL Scheduler GA."""

import functools
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt

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
    _bt_cache: dict[str, float] = field(default=None, init=False, repr=False)
    _tc_cache: dict[str, float] = field(default=None, init=False, repr=False)
    _ov_cache: dict[str, float] = field(default=None, init=False, repr=False)

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

        num_teams = len(all_teams)

        bt_total = 0
        tc_total = 0
        ov_total = 0

        bt_list = []
        tc_list = []
        ov_list = []

        score_lists = [bt_list, tc_list, ov_list]

        for team in all_teams:
            t_bt = self._bt_cache[team.break_time_key()]
            t_tc = self._tc_cache[team.table_consistency_key()]
            t_ov = self._ov_cache[team.opponent_variety_key()]

            team.fitness = (t_bt, t_tc, t_ov)

            bt_total += t_bt
            tc_total += t_tc
            ov_total += t_ov

            bt_list.append(t_bt)
            tc_list.append(t_tc)
            ov_list.append(t_ov)

        totals = (bt_total, tc_total, ov_total)

        # Metric 1: Averages of scores across all teams
        # The higher, the better
        means = [calc_mean(total, num_teams) for total in totals]

        # Metric 2: Coefficient of Variation (CV) for each score, how much variation relative to mean
        # The lower, the better
        _sum_sq_diffs = (
            sum(calc_sum_sq_diffs(x, mean) for x in lst) for lst, mean in zip(score_lists, means, strict=True)
        )
        _std_devs = (calc_std_dev(sum_sq_diff, num_teams) for sum_sq_diff in _sum_sq_diffs)
        _coeff_of_vars = (calc_coeff_of_var(std_dev, mean) for std_dev, mean in zip(_std_devs, means, strict=True))
        ratios = (calc_inversion(coeff) for coeff in _coeff_of_vars)

        # Metric 3: Range of scores for each objective (max - min)
        # The lower, the better
        _ranges = (calc_range(min(lst), max(lst)) if lst else 1 for lst in score_lists)
        range_coeffs = (calc_inversion(range_val) for range_val in _ranges)

        return tuple(
            calc_score(mean, ratio, range_coeff)
            for mean, ratio, range_coeff in zip(means, ratios, range_coeffs, strict=True)
        )


@functools.cache
def calc_mean(total: float, count: int) -> float:
    """Calculate the mean of a total and count."""
    return total / count if count else 0


@functools.cache
def calc_sum_sq_diffs(x: float, mean: float) -> float:
    """Calculate the sum of squared differences."""
    return (x - mean) ** 2


@functools.cache
def calc_std_dev(sum_sq_diff: float, count: int) -> float:
    """Calculate the standard deviation."""
    return sqrt(sum_sq_diff / count) if count else 0


@functools.cache
def calc_coeff_of_var(std_dev: float, mean: float) -> float:
    """Calculate the coefficient of variation."""
    return std_dev / mean if mean else 0


@functools.cache
def calc_range(min_val: float, max_val: float) -> float:
    """Calculate the range coefficient for a value."""
    return max_val - min_val if max_val > min_val else 1


@functools.cache
def calc_inversion(val: float) -> float:
    """Calculate the inversion of a value."""
    return 1 / (1 + val) if val else 1


@functools.cache
def calc_score(mean: float, ratio: float, range_coeff: float) -> float:
    """Calculate the final score based on mean, ratio, and range coefficient."""
    return mean * ratio * range_coeff
