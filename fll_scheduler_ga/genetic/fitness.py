"""Fitness evaluator for the FLL Scheduler GA."""

# import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ..config.config import TournamentConfig
from .schedule import Schedule

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    soft_constraints: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.soft_constraints.extend(
            [
                "BreakTime",
                "OpponentVariety",
                "TableConsistency",
            ]
        )

    def evaluate(self, schedule: Schedule) -> tuple[float, ...] | None:
        """Evaluate the fitness of a schedule."""
        if not schedule:
            logger.debug("%s: %s", "ScheduleExistence", "Schedule is empty")
            return None

        score_map = defaultdict(list)
        for team in schedule.all_teams:
            if rounds_needed := team.rounds_needed():
                logger.debug("%s: %s", "AllEventsScheduled", f"Team {team.identity} needs {rounds_needed} rounds")
                return None

            score_map["BreakTime"].append(team.score_break_time())
            score_map["OpponentVariety"].append(team.score_opponent_variety())
            score_map["TableConsistency"].append(team.score_table_consistency())

        final_scores = []
        for scores in score_map.values():
            mean = sum(scores) / len(scores)
            # sum_sq_diff = sum((x - mean) ** 2 for x in scores)
            # variance = sum_sq_diff / len(scores)
            # stdev = math.sqrt(variance)
            # coeff = stdev / mean if mean > 0 else 0
            # final_scores.append(1.0 / (1.0 + coeff))
            final_scores.append(mean)

        return tuple(final_scores)
