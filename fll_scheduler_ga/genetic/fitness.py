"""Fitness evaluator for the FLL Scheduler GA."""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

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
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)
    score_map: dict[FitnessObjective, float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.score_map = dict.fromkeys(list(FitnessObjective), 0)
        self.objectives.extend(self.score_map.keys())

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
        unique_opponents_possible = self.config.unique_opponents_possible

        if not self.check(schedule):
            return None

        if all_teams := schedule.all_teams():
            score_map = self.score_map.copy()
            for team in all_teams:
                team.score()
                score_map[FitnessObjective.BREAK_TIME] += team.fitness[0]
                score_map[FitnessObjective.TABLE_CONSISTENCY] += team.fitness[1]
                score_map[FitnessObjective.OPPONENT_VARIETY] += team.fitness[2]

            if not (num_teams := len(all_teams)):
                return 1, 1, 1

            if unique_opponents_possible and score_map[FitnessObjective.OPPONENT_VARIETY] / num_teams != 1:
                return tuple(s / (num_teams * 2) for s in score_map.values())
            return tuple(s / num_teams for s in score_map.values())
        return 1, 1, 1
