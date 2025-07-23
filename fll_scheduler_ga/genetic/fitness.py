"""Fitness evaluator for the FLL Scheduler GA."""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache

from ..config.benchmark import BreakTimeFitnessBenchmark, TableConsistencyBenchmark
from ..config.config import TournamentConfig
from .schedule import Schedule

logger: logging.Logger = logging.getLogger(__name__)


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
    break_time_benchmark: BreakTimeFitnessBenchmark
    table_benchmark: TableConsistencyBenchmark
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

        bt_cache = self.break_time_benchmark.cache
        tc_cache = self.table_benchmark.cache["table"]
        ov_cache = self.table_benchmark.cache["opponents"]

        if all_teams := schedule.all_teams():
            if not (num_teams := len(all_teams)):
                return 1, 1, 1

            bt_total = 0
            tc_total = 0
            ov_total = 0
            for team in all_teams:
                bt_total += bt_cache[team.break_time_key()]
                tc_total += tc_cache[team.table_consistency_key()]
                ov_total += ov_cache[team.opponent_variety_key()]

            return (
                get_average(bt_total, num_teams),
                get_average(tc_total, num_teams),
                get_average(ov_total, num_teams),
            )
        return 1, 1, 1


@cache
def get_average(sum_score: float, count: int) -> float:
    """Get the average of a sum_score and count."""
    return sum_score / count if count else 0
