"""Fitness evaluator for the FLL Scheduler GA."""

import logging
from dataclasses import dataclass, field

from ..config.config import TournamentConfig
from .schedule import Schedule

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    objectives: list[str] = field(default_factory=list, init=False)
    score_map: dict[str, list[float]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.score_map = {
            "BreakTime": [],
            "OpponentVariety": [],
            "TableConsistency": [],
        }
        self.objectives.extend(self.score_map.keys())

    def evaluate(self, schedule: Schedule) -> tuple[float, ...] | None:
        """Evaluate the fitness of a schedule."""
        if not schedule:
            logger.debug("%s: %s", "ScheduleExistence", "Schedule is empty")
            return None

        bt_s = 0.0
        ov_s = 0.0
        tc_s = 0.0
        n = 0

        for team in schedule.all_teams():
            if rounds_needed := team.rounds_needed():
                logger.debug("%s: %s", "AllEventsScheduled", f"Team {team.identity} needs {rounds_needed} rounds")
                return None

            bt_s += team.score_break_time()
            ov_s += team.score_opponent_variety()
            tc_s += team.score_table_consistency()

            n += 1

        final_scores = [
            (bt_s / n if n else 0.0),
            (ov_s / n if n else 0.0),
            (tc_s / n if n else 0.0),
        ]

        return tuple(final_scores)
