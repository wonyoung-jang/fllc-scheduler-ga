"""Fitness evaluator for the FLL Scheduler GA."""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from math import sqrt

from ..config.config import TournamentConfig
from ..data_model.team import Team
from .schedule import Schedule

logger: logging.Logger = logging.getLogger(__name__)

PENALTY = 1e-16  # Penalty value for penalizing worse scores


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
        # Configurable bias if completely unique opponens are mathematically possible
        # unique_opponents_possible = self.config.unique_opponents_possible

        if not self.check(schedule):
            return None

        if all_teams := schedule.all_teams():
            if not (num_teams := len(all_teams)):
                return 1, 1, 1

            score_map = self.score_map.copy()
            for team in all_teams:
                score_map[FitnessObjective.BREAK_TIME] += self.score_break_time(team, PENALTY)
                score_map[FitnessObjective.TABLE_CONSISTENCY] += self.score_table_consistency(team, PENALTY)
                score_map[FitnessObjective.OPPONENT_VARIETY] += self.score_opponent_variety(team, PENALTY)

            # Configurable bias if completely unique opponens are mathematically possible
            # if unique_opponents_possible and score_map[FitnessObjective.OPPONENT_VARIETY] / num_teams != 1:
            #     return tuple(s / (num_teams * 2) for s in score_map.values())

            return (
                score_map[FitnessObjective.BREAK_TIME] / num_teams,
                score_map[FitnessObjective.TABLE_CONSISTENCY] / num_teams,
                score_map[FitnessObjective.OPPONENT_VARIETY] / num_teams,
            )
        return 1, 1, 1

    def score_break_time(self, team: Team, penalty: float) -> float:
        """Calculate a score based on the break times between events."""
        if len(team.events) < 2:
            return 1

        _events = sorted(team.events, key=lambda e: e.identity)
        break_times = []

        for i in range(1, len(_events)):
            start = _events[i].timeslot.start
            stop = _events[i - 1].timeslot.stop
            duration_seconds = (start - stop).total_seconds()
            break_times.append(duration_seconds // 60)

        if not (n := len(break_times)):
            return 1

        mean_x = sum(break_times) / n
        if mean_x == 0:
            return 0

        sum_sq_diff = sum((b - mean_x) ** 2 for b in break_times)
        zeros = break_times.count(0)
        return calc_break_time(mean_x, sum_sq_diff, n, zeros, penalty)

    def score_table_consistency(self, team: Team, penalty: float) -> float:
        """Calculate a score based on the consistency of table assignments."""
        locations = [e.location for e in team.events if e.location.teams_per_round == 2]
        unique, total = self._get_unique_and_total(locations)
        return calc_table_consistency(unique, total, penalty)

    def score_opponent_variety(self, team: Team, penalty: float) -> float:
        """Calculate a score based on the variety of opponents faced."""
        unique, total = self._get_unique_and_total(team.opponents)
        return calc_opponent_variety(unique, total, penalty)

    @staticmethod
    def _get_unique_and_total(items: list[int]) -> tuple[int, int]:
        """Get the count of unique and total items."""
        return len(set(items)), len(items)


@cache
def calc_break_time(mean: float, sum_sq_diff: float, n: int, zeros: int, penalty: float) -> float:
    """Calculate a score based on the break times between events."""
    coefficient = sqrt(sum_sq_diff / n) / mean  # The coefficient is the standard deviation normalized by the mean
    ratio = 1 / (1 + coefficient)  # The lower the coefficient, the better
    b_penalty = penalty**zeros
    return ratio * b_penalty


@cache
def calc_table_consistency(unique: int, total: int, penalty: float) -> float:
    """Calculate a score based on the consistency of table assignments."""
    ratio = unique / total if total else 1
    ratio = 1 / (1 + ratio)  # The lower the number of unique assignments, the better
    t_penalty = penalty**total if unique == total else 1
    return ratio * t_penalty


@cache
def calc_opponent_variety(unique: int, total: int, penalty: float) -> float:
    """Calculate a score based on the variety of opponents faced."""
    ratio = unique / total if total else 1
    o_penalty = penalty ** (total - unique) if unique != total else 1
    return ratio * o_penalty
