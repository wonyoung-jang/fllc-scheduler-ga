"""Fitness evaluator for the FLL Scheduler GA."""

import logging
import statistics
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

        bt_s = []
        ov_s = []
        tc_s = []

        all_teams = schedule.all_teams()

        for team in all_teams:
            if team.rounds_needed():
                logger.debug("%s: %s", HardConstraints.ALL_EVENTS_SCHEDULED, f"{team.identity} team needs rounds")
                return None

            bt_s.append(team.score_break_time())
            ov_s.append(team.score_opponent_variety())
            tc_s.append(team.score_table_consistency())

        bt_ndist = statistics.NormalDist.from_samples(bt_s)
        ov_ndist = statistics.NormalDist.from_samples(ov_s)
        tc_ndist = statistics.NormalDist.from_samples(tc_s)
        bt_score = bt_ndist.mean * (1 - bt_ndist.stdev)
        ov_score = ov_ndist.mean * (1 - ov_ndist.stdev)
        tc_score = tc_ndist.mean * (1 - tc_ndist.stdev)

        return bt_score, ov_score, tc_score
