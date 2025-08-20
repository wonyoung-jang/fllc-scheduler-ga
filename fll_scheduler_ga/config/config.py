"""Configuration for the tournament scheduler GA."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import getLogger
from math import ceil

from ..data_model.location import Location

logger = getLogger(__name__)

type RoundType = str


@dataclass(slots=True, frozen=True)
class Round:
    """Representation of a round in the FLL tournament."""

    roundtype: RoundType
    rounds_per_team: int
    teams_per_round: int
    times: list[datetime]
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    num_teams: int
    location: str
    locations: list[Location]

    def __post_init__(self) -> None:
        """Post-initialization to validate the round configuration."""
        logger.debug("Round configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the Round."""
        return (
            f"\n\tRound:"
            f"\n\t  roundtype        : {self.roundtype}"
            f"\n\t  teams_per_round  : {self.teams_per_round}"
            f"\n\t  rounds_per_team  : {self.rounds_per_team}"
            f"\n\t  times            : {self.times}"
            f"\n\t  start_time       : {self.start_time}"
            f"\n\t  stop_time        : {self.stop_time}"
            f"\n\t  duration_minutes : {self.duration_minutes}"
            f"\n\t  num_timeslots    : {self.get_num_slots()}"
            f"\n\t  num_teams        : {self.num_teams}"
            f"\n\t  location         : {self.location}"
            f"\n\t  locations        : {self.locations}"
        )

    def get_num_slots(self) -> int:
        """Get the number of slots available for this round."""
        if self.times:
            return len(self.times)

        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = len(self.locations)

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
    time_fmt: str
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
        rounds_str = ", ".join(f"{r.roundtype}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())

        return (
            f"\n\tTournamentConfig:"
            f"\n\t  num_teams                 : {self.num_teams}"
            f"\n\t  time_fmt                  : {self.time_fmt}"
            f"\n\t  rounds                    : {rounds_str}"
            f"\n\t  round_requirements        : {round_reqs_str}"
            f"\n\t  total_slots               : {self.total_slots}"
            f"\n\t  unique_opponents_possible : {self.unique_opponents_possible}"
            f"\n\t  weights                   : {self.weights}"
        )
