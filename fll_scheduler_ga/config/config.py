"""Configuration for the tournament scheduler GA."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import getLogger
from math import ceil

logger = getLogger(__name__)

type RoundType = str


@dataclass(slots=True, frozen=True)
class Round:
    """Representation of a round in the FLL tournament."""

    round_type: RoundType
    rounds_per_team: int
    teams_per_round: int
    times: list[datetime]
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    num_locations: int
    num_teams: int

    def get_num_slots(self) -> int:
        """Get the number of slots available for this round."""
        if self.times:
            return len(self.times)

        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = self.num_locations * self.teams_per_round

        if slots_per_timeslot == 0:
            return 0

        minimum_slots = ceil(total_num_teams / slots_per_timeslot)

        if self.stop_time:
            total_available = self.stop_time - self.start_time
            slots_in_window = int(total_available / self.duration_minutes)
            return max(minimum_slots, slots_in_window)

        return minimum_slots


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    total_slots: int
    unique_opponents_possible: bool
    weights: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        logger.debug("Tournament configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.round_type}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())

        return (
            f"TournamentConfig:\n"
            f"\tNumber of Teams: {self.num_teams}\n"
            f"\tRound Types: {rounds_str}\n"
            f"\tRound Requirements: {round_reqs_str}\n"
            f"\tTotal Slots: {self.total_slots}\n"
            f"\tUnique Opponents Possible: {self.unique_opponents_possible}\n"
            f"\tWeights: {self.weights}"
        )
