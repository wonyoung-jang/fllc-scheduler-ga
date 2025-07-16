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
    OPPONENT_VARIETY = "OpponentVariety"
    TABLE_CONSISTENCY = "TableConsistency"


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)
    score_map: dict[FitnessObjective, list[float]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.score_map = {key: [] for key in list(FitnessObjective)}
        self.objectives.extend(self.score_map.keys())

    def evaluate(self, schedule: Schedule) -> tuple[float, ...] | None:
        """Evaluate the fitness of a schedule."""
        if not schedule:
            logger.debug("%s: %s", HardConstraints.SCHEDULE_EXISTENCE, "Schedule is empty")
            return None

        if not schedule.all_teams_scheduled():
            logger.debug("%s: %s", HardConstraints.ALL_EVENTS_SCHEDULED, "Not all events are scheduled")
            return None

        bt_s = 0
        ov_s = 0
        tc_s = 0

        all_teams = schedule.all_teams()

        if not all_teams:
            return 1, 1, 1

        for team in all_teams:
            team.score()
            bt_s += team.fitness[0]
            ov_s += team.fitness[1]
            tc_s += team.fitness[2]

        num_teams = len(all_teams)

        bt_score = bt_s / num_teams
        ov_score = ov_s / num_teams
        tc_score = tc_s / num_teams

        return bt_score, ov_score, tc_score
