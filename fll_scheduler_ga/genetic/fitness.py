"""Fitness evaluator for the FLL Scheduler GA."""

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

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(
            [
                FitnessObjective.BREAK_TIME,
                FitnessObjective.TABLE_CONSISTENCY,
                FitnessObjective.OPPONENT_VARIETY,
            ]
        )

    def check(self, schedule: Schedule) -> bool:
        """Check if the schedule meets hard constraints.

        Args:
            schedule (Schedule): The schedule to check.

        Returns:
            bool: True if the schedule meets all hard constraints, False otherwise.

        """
        if not schedule:
            logger.debug("%s: %s", HardConstraints.SCHEDULE_EXISTENCE, "Schedule is empty")
            return False

        if len(schedule) < self.config.total_slots:
            logger.debug("%s: %s", HardConstraints.ALL_EVENTS_SCHEDULED, "Not all events are scheduled")
            return False

        return True

    def evaluate(self, schedule: Schedule) -> tuple[float, ...] | None:
        """Evaluate the fitness of a schedule.

        Args:
            schedule (Schedule): The schedule to evaluate.

        Returns:
            tuple[float, ...] | None: A tuple of fitness scores for each objective,

        """
        if not self.check(schedule):
            return None

        _benchmark = self.benchmark
        bt_cache = _benchmark.timeslots
        tc_cache = _benchmark.table
        ov_cache = _benchmark.opponents

        if all_teams := schedule.all_teams():
            if not (num_teams := len(all_teams)):
                return 1, 1, 1

            bt_total = 0
            tc_total = 0
            ov_total = 0

            bt_list = []
            tc_list = []
            ov_list = []

            for team in all_teams:
                t_bt = bt_cache[team.break_time_key()]
                t_tc = tc_cache[team.table_consistency_key()]
                t_ov = ov_cache[team.opponent_variety_key()]
                team.fitness = (t_bt, t_tc, t_ov)
                bt_total += t_bt
                tc_total += t_tc
                ov_total += t_ov
                bt_list.append(t_bt)
                tc_list.append(t_tc)
                ov_list.append(t_ov)

            mean_bt = bt_total / num_teams
            mean_tc = tc_total / num_teams
            mean_ov = ov_total / num_teams

            sum_sq_diff_bt = sum((bt - mean_bt) ** 2 for bt in bt_list)
            sum_sq_diff_tc = sum((tc - mean_tc) ** 2 for tc in tc_list)
            sum_sq_diff_ov = sum((ov - mean_ov) ** 2 for ov in ov_list)

            std_dev_bt = sqrt(sum_sq_diff_bt / num_teams)
            std_dev_tc = sqrt(sum_sq_diff_tc / num_teams)
            std_dev_ov = sqrt(sum_sq_diff_ov / num_teams)

            coeff_bt = std_dev_bt / mean_bt if mean_bt else 0
            coeff_tc = std_dev_tc / mean_tc if mean_tc else 0
            coeff_ov = std_dev_ov / mean_ov if mean_ov else 0

            ratio_bt = 1 / (1 + coeff_bt)
            ratio_tc = 1 / (1 + coeff_tc)
            ratio_ov = 1 / (1 + coeff_ov)

            range_bt = max(bt_list) - min(bt_list) if bt_list else 1
            range_tc = max(tc_list) - min(tc_list) if tc_list else 1
            range_ov = max(ov_list) - min(ov_list) if ov_list else 1

            range_coeff_bt = 1 / (1 + range_bt)
            range_coeff_tc = 1 / (1 + range_tc)
            range_coeff_ov = 1 / (1 + range_ov)

            return (
                ratio_bt * mean_bt * range_coeff_bt,
                ratio_tc * mean_tc * range_coeff_tc,
                ratio_ov * mean_ov * range_coeff_ov,
            )
        return 1, 1, 1
