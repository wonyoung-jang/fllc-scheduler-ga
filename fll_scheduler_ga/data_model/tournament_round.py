"""Configuration for a tournament round."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from .location import Location
    from .time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TournamentRound:
    """Representation of a round in the FLL tournament."""

    roundtype: str
    roundtype_idx: int
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
        logger.debug("TournamentRound configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the TournamentRound."""
        return (
            f"\n\tRound:"
            f"\n\t  roundtype        : {self.roundtype}"
            f"\n\t  roundtype_idx    : {self.roundtype_idx}"
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
