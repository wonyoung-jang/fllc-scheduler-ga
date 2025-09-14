"""Configuration for the tournament scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from math import ceil
from typing import TYPE_CHECKING

from ..data_model.time import TimeSlot

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from ..data_model.location import Location

logger = getLogger(__name__)

type RoundType = str


@dataclass(slots=True)
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
    num_timeslots: int = field(default=None)
    timeslots: list[TimeSlot] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization to validate the round configuration."""
        self.num_timeslots = self.calc_num_timeslots()
        self.init_timeslots()
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
            f"\n\t  num_timeslots    : {self.num_timeslots}"
            f"\n\t  num_teams        : {self.num_teams}"
            f"\n\t  location         : {self.location}"
            f"\n\t  locations        : {[str(loc) for loc in self.locations]}"
            f"\n\t  timeslots        : {[str(ts) for ts in self.timeslots]}"
        )

    def calc_num_timeslots(self) -> int:
        """Initialize the number of timeslots for the round."""
        if self.times:
            return len(self.times)

        if not self.locations:
            return 0

        total_teams = self.num_teams * self.rounds_per_team
        return ceil(total_teams / len(self.locations))

    def init_timeslots(self) -> None:
        """Initialize the timeslots for the round."""
        times = self.times
        duration = self.duration_minutes
        start = times[0] if times else self.start_time
        for i in range(1, self.num_timeslots + 1):
            if not times:
                stop = start + duration
            elif i < len(times):
                stop = times[i]
            elif i == len(times):
                stop += duration

            self.timeslots.append(TimeSlot(start, stop))
            start = stop

        self.init_start_time()
        self.init_stop_time()

    def init_start_time(self) -> None:
        """Initialize the start time if not provided."""
        if self.times:
            self.start_time = self.times[0]
        else:
            self.start_time = self.timeslots[0].start

    def init_stop_time(self) -> None:
        """Initialize the stop time if not provided."""
        if self.times:
            self.stop_time = self.times[-1] + self.duration_minutes
        else:
            self.stop_time = self.timeslots[-1].stop


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    time_fmt: str
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    total_slots: int
    unique_opponents_possible: bool
    weights: tuple[float, float]

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
