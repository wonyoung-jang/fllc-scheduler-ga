"""Dataclass models for application configuration."""

import datetime as dt
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fll_scheduler_ga.constants import DATA_MODEL_VERSION, FITNESS_MODEL_VERSION

if TYPE_CHECKING:
    from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

DEFAULT_DT = dt.datetime.min.replace(tzinfo=dt.UTC)
ASCII_OFFSET = 64


@dataclass(slots=True)
class ScheduleContext:
    """Holds class-level context for Schedule instances."""

    conflict_map: dict[int, set[int]]
    event_props: EventProperties
    teams_list: np.ndarray
    teams_roundreqs_arr: np.ndarray
    empty_schedule: np.ndarray


@dataclass(slots=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    schedule: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness: np.ndarray = field(default_factory=lambda: np.array([]))
    team_fitnesses: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = -1
    origin: str = "Builder"
    mutations: int = 0
    clones: int = 0
    _hash: int | None = None
    team_events: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    team_rounds: np.ndarray = field(default_factory=lambda: np.array([]))
    # Class variables
    ctx: ClassVar[ScheduleContext]

    def __post_init__(self) -> None:
        """Post-initialization to set up fitness array."""
        if self.schedule.size == 0:
            self.schedule = Schedule.ctx.empty_schedule.copy()
        if self.team_rounds.size == 0:
            self.team_rounds = Schedule.ctx.teams_roundreqs_arr.copy()

    def __len__(self) -> int | np.signedinteger:
        """Return the number of scheduled events."""
        return np.count_nonzero(self.schedule >= 0)

    def get_size(self) -> int | np.signedinteger:
        """Return the number of scheduled events."""
        return self.__len__()

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._hash is None:
            self._hash = hash(frozenset(frozenset(events) for events in self.team_events.values()))
        return self._hash

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        return Schedule(
            schedule=self.schedule.copy(),
            fitness=self.fitness.copy(),
            team_fitnesses=self.team_fitnesses.copy(),
            rank=self.rank,
            origin=self.origin,
            mutations=self.mutations,
            clones=self.clones + 1,
            _hash=self._hash,
            team_events={k: v.copy() for k, v in self.team_events.items()},
            team_rounds=self.team_rounds.copy(),
        )

    def swap_assignment(self, team: int, old_event: int, new_event: int) -> None:
        """Switch an event for a team in the schedule."""
        if team == -1:
            return
        self.unassign(team, old_event)
        self.assign(team, new_event)

    def assign(self, team: int, event: int) -> None:
        """Add an event to a team's scheduled events."""
        if team == -1:
            return
        roundtype = Schedule.ctx.event_props.roundtype_idx[event]
        self.team_events[team].add(event)
        self.team_rounds[team, roundtype] -= 1
        self.schedule[event] = team
        self._hash = None

    def unassign(self, team: int, event: int) -> None:
        """Remove an event from a team's scheduled events."""
        if team == -1:
            return
        roundtype = Schedule.ctx.event_props.roundtype_idx[event]
        self.team_events[team].remove(event)
        self.team_rounds[team, roundtype] += 1
        self.schedule[event] = -1
        self._hash = None

    def needs_round(self, team: int, roundtype: int) -> bool:
        """Check if a team still needs to participate in a given round type."""
        return self.team_rounds[team, roundtype] > 0

    def all_rounds_needed(self, roundtype: int) -> np.ndarray:
        """Return all teams that still need roundtype."""
        return (self.team_rounds[:, roundtype] > 0).nonzero()[0]

    def any_rounds_needed(self) -> bool:
        """Check if any team still needs rounds."""
        return self.team_rounds.sum() > 0

    def conflicts(self, team: int, new_event: int, *, ignore: int | None = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            team (int): The team to check for conflicts.
            new_event (int): The new event to check for conflicts.
            ignore (int): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        if team == -1:
            return False
        events_to_check = self.team_events[team]
        if ignore is not None and ignore in events_to_check:
            events_to_check = events_to_check - {ignore}
        if new_event in events_to_check:
            return True
        return not events_to_check.isdisjoint(Schedule.ctx.conflict_map[new_event])

    def scheduled_events(self) -> np.ndarray:
        """Return the indices of scheduled events."""
        return (self.schedule >= 0).nonzero()[0]

    def unscheduled_events(self) -> np.ndarray:
        """Return the indices of unscheduled events."""
        return (self.schedule == -1).nonzero()[0]


@dataclass(slots=True, frozen=True)
class TimeSlot:
    """Data model for a time slot in the FLL Scheduler GA."""

    idx: int = 0
    start: dt.datetime = DEFAULT_DT
    stop_active: dt.datetime = DEFAULT_DT
    stop_cycle: dt.datetime = DEFAULT_DT
    time_fmt: ClassVar[str]

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        return f"{self.start.strftime(TimeSlot.time_fmt)}-{self.stop_cycle.strftime(TimeSlot.time_fmt)}"

    def __lt__(self, other: TimeSlot) -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: TimeSlot) -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop_cycle and other.start < self.stop_cycle


@dataclass(slots=True)
class Location:
    """Data model for a location in the FLL Scheduler GA."""

    idx: int = 0
    locationtype: str = "Null"
    name: int = 1
    side: int = -1
    teams_per_round: int = 1
    _str: str = ""
    _hash: int = 0

    def __post_init__(self) -> None:
        """Post-initialization to set private attributes."""
        ltr_id = chr(ASCII_OFFSET + self.name)
        if self.side > 0:
            self._str = f"{self.locationtype} {ltr_id}{self.side}"
        else:
            self._str = f"{self.locationtype} {ltr_id}"
        self._hash = hash((self.name, self.side))

    def __str__(self) -> str:
        """Represent the Location as a string."""
        return self._str

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        return self._hash


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    idx: int = 0
    roundtype: str = "Null"
    roundtype_idx: int = 0
    timeslot: TimeSlot = field(default_factory=TimeSlot)
    location: Location = field(default_factory=Location)
    paired: int = -1
    conflicts: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.idx}, {self.roundtype}, {self.location}, {self.timeslot}"

    def pair(self, other: Event) -> None:
        """Pair this event with another event."""
        self.paired = other.idx
        other.paired = self.idx


@dataclass(slots=True)
class TournamentRound:
    """Representation of a round in the FLL tournament."""

    roundtype: str
    roundtype_idx: int
    rounds_per_team: int
    teams_per_round: int
    times: tuple[datetime, ...]
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    location_type: str
    locations: tuple[Location, ...]
    num_timeslots: int
    timeslots: tuple[TimeSlot, ...]
    slots_total: int
    slots_required: int
    slots_empty: int
    unfilled_allowed: bool

    @property
    def canonical_tuple(self) -> tuple[Any, ...]:
        """Return a canonical tuple representation of the configuration."""
        return tuple(
            getattr(self, s) if not isinstance(getattr(self, s), tuple) else frozenset(getattr(self, s))
            for s in self.__slots__
        )


@dataclass(slots=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    time_fmt: str
    rounds: tuple[TournamentRound, ...]
    roundreqs: dict[str, int]
    round_idx_to_tpr: dict[int, int]
    total_slots_required: int
    unique_opponents_possible: bool
    max_events_per_team: int
    all_locations: tuple[Location, ...]
    all_timeslots: tuple[TimeSlot, ...]
    is_interleaved: bool

    def __hash__(self) -> int:
        """Return hash of TournamentConfig."""
        return hash(
            getattr(self, s) if not isinstance(getattr(self, s), dict) else tuple(sorted(getattr(self, s).items()))
            for s in self.__slots__
        )

    @property
    def canonical_round_tuples(self) -> tuple[tuple[Any, ...], ...]:
        """Return canonical tuple representations of all rounds."""
        return tuple(r.canonical_tuple for r in self.rounds)

    @property
    def canonical_roundreqs_tuple(self) -> tuple[tuple[str, int], ...]:
        """Return canonical tuple representation of round requirements."""
        return tuple(sorted(self.roundreqs.items()))

    @property
    def n_total_events(self) -> int:
        """Return the total number of events possible in the tournament."""
        return sum(r.slots_total for r in self.rounds)


@dataclass(slots=True)
class EventRepository:
    """Class to hold event-related data."""

    events: tuple[Event, ...]
    events_idx: np.ndarray
    singles_or_side1_idx: np.ndarray
    conflict_map: dict[int, set[int]]
    mapping: dict[int, Event]
    roundtypes: dict[int, list[int]]
    timeslots: dict[tuple[int, int], list[int]]
    matches: dict[int, list[tuple[int, int]]]


@dataclass(slots=True)
class EventProperties:
    """Holds properties of an event for fast access during evaluation."""

    roundtype: np.ndarray
    roundtype_idx: np.ndarray
    timeslot: np.ndarray
    timeslot_idx: np.ndarray
    start: np.ndarray
    stop_active: np.ndarray
    stop_cycle: np.ndarray
    location: np.ndarray
    loc_str: np.ndarray
    loc_type: np.ndarray
    loc_idx: np.ndarray
    loc_name: np.ndarray
    loc_side: np.ndarray
    teams_per_round: np.ndarray
    paired_idx: np.ndarray


@dataclass(slots=True)
class GASeedData:
    """GA seed data object."""

    config: TournamentConfig | None = None
    population: list[Schedule] = field(default_factory=list)
    version: int = DATA_MODEL_VERSION


@dataclass(slots=True)
class BenchmarkSeedData:
    """Seed data object for fitness benchmarks."""

    opponents: np.ndarray
    best_timeslot_score: float
    version: int = FITNESS_MODEL_VERSION

    def __bool__(self) -> bool:
        """Validate loaded benchmark data."""
        if self.version != FITNESS_MODEL_VERSION:
            logger.warning(
                "Benchmark version mismatch: Expected %d, found %d. Recalculating benchmarks.",
                FITNESS_MODEL_VERSION,
                self.version,
            )
            return False
        return True
