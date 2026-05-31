"""Dataclass models for application configuration."""

import datetime as dt
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fll_scheduler_ga.constants import DATA_MODEL_VERSION, FITNESS_MODEL_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime, timedelta

    from fll_scheduler_ga.domain.schedule import Schedule


logger = logging.getLogger(__name__)

DEFAULT_DT = dt.datetime.min.replace(tzinfo=dt.UTC)
ASCII_OFFSET = 64


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
        return (
            self.roundtype,
            self.roundtype_idx,
            self.rounds_per_team,
            self.teams_per_round,
            frozenset(self.times),
            self.start_time,
            self.stop_time,
            self.duration_minutes,
            self.location_type,
            frozenset(self.locations),
            self.num_timeslots,
            frozenset(self.timeslots),
            self.slots_total,
            self.slots_required,
            self.slots_empty,
            self.unfilled_allowed,
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

    def __eq__(self, other: object) -> bool:
        """Check equality between two TournamentConfig instances."""
        if not isinstance(other, TournamentConfig):
            return NotImplemented
        return (
            self.num_teams == other.num_teams
            and self.time_fmt == other.time_fmt
            and self.rounds == other.rounds
            and self.roundreqs == other.roundreqs
            and self.round_idx_to_tpr == other.round_idx_to_tpr
            and self.total_slots_required == other.total_slots_required
            and self.unique_opponents_possible == other.unique_opponents_possible
            and self.all_locations == other.all_locations
            and self.all_timeslots == other.all_timeslots
            and self.max_events_per_team == other.max_events_per_team
            and self.is_interleaved == other.is_interleaved
        )

    def __hash__(self) -> int:
        """Return hash of TournamentConfig."""
        return hash(
            (
                self.num_teams,
                self.time_fmt,
                self.rounds,
                tuple(sorted(self.roundreqs.items())),
                tuple(sorted(self.round_idx_to_tpr.items())),
                self.total_slots_required,
                self.unique_opponents_possible,
                self.max_events_per_team,
                self.all_locations,
                self.all_timeslots,
                self.is_interleaved,
            )
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
class EventFactory:
    """Factory class to create Events based on TournamentRound configurations."""

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


def generate_events_from_round(r: TournamentRound, event_idx_iter: Iterator[int]) -> Iterator[Event]:
    """Generate all possible Events for a given TournamentRound configuration.

    Args:
        r (TournamentRound): The tournament round configuration to generate events for.
        event_idx_iter (Iterator[int]): An iterator to generate unique event IDs.

    Yields:
        Event: An event for the round with a time slot and a location.

    """
    for ts in r.timeslots:
        if r.teams_per_round == 1:
            for loc in r.locations:
                event = Event(
                    idx=next(event_idx_iter),
                    roundtype=r.roundtype,
                    roundtype_idx=r.roundtype_idx,
                    timeslot=ts,
                    location=loc,
                )
                yield event
        elif r.teams_per_round == 2:
            event1 = Event()
            for loc in r.locations:
                if loc.side == 1:
                    event1 = Event(
                        idx=next(event_idx_iter),
                        roundtype=r.roundtype,
                        roundtype_idx=r.roundtype_idx,
                        timeslot=ts,
                        location=loc,
                    )
                elif loc.side == 2:
                    event2 = Event(
                        idx=next(event_idx_iter),
                        roundtype=r.roundtype,
                        roundtype_idx=r.roundtype_idx,
                        timeslot=ts,
                        location=loc,
                    )
                    event1.pair(event2)
                    yield from (event1, event2)


def build_event_factory(rounds: tuple[TournamentRound, ...]) -> EventFactory:
    """Build an EventFactory from the tournament configuration."""
    event_idx_iter = itertools.count()
    events = tuple(e for r in rounds for e in generate_events_from_round(r, event_idx_iter))
    events_idx = np.array([e.idx for e in events], dtype=int)
    singles_or_side1 = tuple(e for e in events if e.paired == -1 or (e.paired != -1 and e.location.side == 1))
    singles_or_side1_idx = np.array([e.idx for e in singles_or_side1], dtype=int)
    timeslots = defaultdict(list)
    matches = defaultdict(list)
    roundtypes = defaultdict(list)
    for e in events:
        timeslots[(e.roundtype_idx, e.timeslot.idx)].append(e.idx)
        roundtypes[e.roundtype_idx].append(e.idx)
    for e in singles_or_side1:
        if e.paired != -1:
            matches[e.roundtype_idx].append((e.idx, e.paired))
    n = len(events)
    c_matrix = np.full((n, n), fill_value=False, dtype=bool)
    for e1, e2 in itertools.combinations(events, 2):
        if e1.timeslot.overlaps(e2.timeslot):
            e1.conflicts.append(e2.idx)
            e2.conflicts.append(e1.idx)
            c_matrix[e1.idx, e2.idx] = True
            c_matrix[e2.idx, e1.idx] = True
    for e in events:
        e.conflicts = sorted(set(e.conflicts))
        logger.debug("%s has %d conflicts: %s", e, len(e.conflicts), e.conflicts)
    for i in range(n):
        c_matrix[i, i] = True  # An event conflicts with itself
    mapping = {e.idx: e for e in events}
    conflict_map = {e.idx: set(e.conflicts) for e in events}
    return EventFactory(
        events=events,
        events_idx=events_idx,
        singles_or_side1_idx=singles_or_side1_idx,
        conflict_map=conflict_map,
        mapping=mapping,
        roundtypes=roundtypes,
        timeslots=timeslots,
        matches=matches,
    )


def build_event_props(event_map: dict[int, Event]) -> EventProperties:
    """Build EventProperties from an event mapping."""
    ep_dtype = np.dtype(
        [
            ("roundtype", "U50"),
            ("roundtype_idx", int),
            ("timeslot", object),
            ("timeslot_idx", int),
            ("start", int),
            ("stop_active", int),
            ("stop_cycle", int),
            ("location", object),
            ("loc_str", "U50"),
            ("loc_type", "U50"),
            ("loc_idx", int),
            ("loc_name", int),
            ("loc_side", int),
            ("teams_per_round", int),
            ("paired_idx", int),
        ]
    )
    ep: np.ndarray = np.zeros(len(event_map), dtype=ep_dtype)
    for i, e in event_map.items():
        ep[i]["roundtype"] = e.roundtype
        ep[i]["roundtype_idx"] = e.roundtype_idx
        ep[i]["timeslot"] = e.timeslot
        ep[i]["timeslot_idx"] = e.timeslot.idx
        ep[i]["start"] = int(e.timeslot.start.timestamp())
        ep[i]["stop_active"] = int(e.timeslot.stop_active.timestamp())
        ep[i]["stop_cycle"] = int(e.timeslot.stop_cycle.timestamp())
        ep[i]["location"] = e.location
        ep[i]["loc_str"] = str(e.location)
        ep[i]["loc_type"] = e.location.locationtype
        ep[i]["loc_idx"] = e.location.idx
        ep[i]["loc_name"] = e.location.name
        ep[i]["loc_side"] = e.location.side
        ep[i]["teams_per_round"] = e.location.teams_per_round
        ep[i]["paired_idx"] = e.paired
    return EventProperties(
        roundtype=ep["roundtype"],
        roundtype_idx=ep["roundtype_idx"],
        timeslot=ep["timeslot"],
        timeslot_idx=ep["timeslot_idx"],
        start=ep["start"],
        stop_active=ep["stop_active"],
        stop_cycle=ep["stop_cycle"],
        location=ep["location"],
        loc_str=ep["loc_str"],
        loc_type=ep["loc_type"],
        loc_idx=ep["loc_idx"],
        loc_name=ep["loc_name"],
        loc_side=ep["loc_side"],
        teams_per_round=ep["teams_per_round"],
        paired_idx=ep["paired_idx"],
    )
