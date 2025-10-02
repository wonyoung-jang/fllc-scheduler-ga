"""Configuration for the tournament scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from .location import Location
    from .time import TimeSlot

logger = getLogger(__name__)

type RoundType = str


@dataclass(slots=True, frozen=True)
class RoundTypeClass:
    """Class to represent a round type."""

    idx: int
    name: str

    def __str__(self) -> str:
        """Represent the RoundTypeClass."""
        return f"{self.name}({self.idx})"


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
    num_timeslots: int
    timeslots: list[TimeSlot]

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
            f"\n\t  num_teams        : {self.num_teams}"
            f"\n\t  location         : {self.location}"
            f"\n\t  locations        : {[str(loc) for loc in self.locations]}"
            f"\n\t  num_timeslots    : {self.num_timeslots}"
            f"\n\t  timeslots        : {[str(ts) for ts in self.timeslots]}"
        )


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    time_fmt: str
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    round_to_int: dict[RoundType, int]
    round_to_tpr: dict[RoundType, int]
    total_slots_possible: int
    total_slots_required: int
    unique_opponents_possible: bool
    weights: tuple[float, float]
    locations: list[Location]

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
            f"\n\t  round_to_int              : {self.round_to_int}"
            f"\n\t  round_to_tpr              : {self.round_to_tpr}"
            f"\n\t  total_slots_possible      : {self.total_slots_possible}"
            f"\n\t  total_slots_required      : {self.total_slots_required}"
            f"\n\t  unique_opponents_possible : {self.unique_opponents_possible}"
            f"\n\t  weights                   : {self.weights}"
            f"\n\t  locations                 : {[str(loc) for loc in self.locations]}"
        )
