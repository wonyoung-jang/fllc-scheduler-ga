"""Dataclass models for application configuration."""

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.domain.event import Event
from fll_scheduler_ga.domain.location import Location
from fll_scheduler_ga.domain.timeslot import TimeSlot

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from datetime import datetime, timedelta

    from fll_scheduler_ga.domain.location import Location

logger = logging.getLogger(__name__)


def are_rounds_overlapping(rounds: Iterable[TournamentRound]) -> bool:
    """Check if any rounds are interleaved in time."""
    _starts = (r.start_time for r in rounds)
    _stops = (r.stop_time for r in rounds)
    timeslots = tuple(
        TimeSlot(idx=0, start=start, stop_active=stop_cycle, stop_cycle=stop_cycle)
        for start, stop_cycle in zip(_starts, _stops, strict=True)
    )
    return any(timeslots[i].overlaps(timeslots[j]) for i in range(len(timeslots)) for j in range(i + 1, len(timeslots)))


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

    def get_canonical_tuple(self) -> tuple[Any, ...]:
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

    def create_events(self, event_idx_iter: Iterator[int]) -> Iterator[Event]:
        """Generate all possible Events for a given TournamentRound configuration.

        Args:
            event_idx_iter (Iterator[int]): An iterator to generate unique event IDs.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        for ts in self.timeslots:
            if self.teams_per_round == 1:
                for loc in self.locations:
                    event = Event(
                        idx=next(event_idx_iter),
                        roundtype=self.roundtype,
                        roundtype_idx=self.roundtype_idx,
                        timeslot=ts,
                        location=loc,
                    )
                    yield event
            elif self.teams_per_round == 2:
                event1 = Event()
                for loc in self.locations:
                    if loc.side == 1:
                        event1 = Event(
                            idx=next(event_idx_iter),
                            roundtype=self.roundtype,
                            roundtype_idx=self.roundtype_idx,
                            timeslot=ts,
                            location=loc,
                        )
                    elif loc.side == 2:
                        event2 = Event(
                            idx=next(event_idx_iter),
                            roundtype=self.roundtype,
                            roundtype_idx=self.roundtype_idx,
                            timeslot=ts,
                            location=loc,
                        )
                        event1.pair(event2)
                        yield from (event1, event2)


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

    def get_canonical_round_tuples(self) -> tuple[tuple[Any, ...], ...]:
        """Return canonical tuple representations of all rounds."""
        return tuple(r.get_canonical_tuple() for r in self.rounds)

    def get_canonical_roundreqs_tuple(self) -> tuple[tuple[str, int], ...]:
        """Return canonical tuple representation of round requirements."""
        return tuple(sorted(self.roundreqs.items()))

    def get_n_total_events(self) -> int:
        """Return the total number of events possible in the tournament."""
        return sum(r.slots_total for r in self.rounds)


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on TournamentRound configurations."""

    config: TournamentConfig
    events: tuple[Event, ...] = field(init=False)
    events_idx: np.ndarray = field(init=False)
    singles_or_side1_idx: np.ndarray = field(init=False)
    conflict_map: dict[int, set[int]] = field(init=False)
    mapping: dict[int, Event] = field(init=False)
    roundtypes: dict[int, list[int]] = field(init=False)
    timeslots: dict[tuple[int, int], list[int]] = field(init=False)
    matches: dict[int, list[tuple[int, int]]] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        event_idx_iter = itertools.count()
        self.events = tuple(e for r in self.config.rounds for e in r.create_events(event_idx_iter))
        self.events_idx = np.array([e.idx for e in self.events], dtype=int)
        _singles_or_side1 = tuple(e for e in self.events if e.paired == -1 or (e.paired != -1 and e.location.side == 1))
        self.singles_or_side1_idx = np.array([e.idx for e in _singles_or_side1], dtype=int)
        self.timeslots = defaultdict(list)
        self.matches = defaultdict(list)
        self.roundtypes = defaultdict(list)
        for e in self.events:
            self.timeslots[(e.roundtype_idx, e.timeslot.idx)].append(e.idx)
            self.roundtypes[e.roundtype_idx].append(e.idx)
        for e in _singles_or_side1:
            if e.paired != -1:
                self.matches[e.roundtype_idx].append((e.idx, e.paired))
        for e1, e2 in itertools.combinations(self.events, 2):
            if e1.timeslot.overlaps(e2.timeslot):
                e1.conflicts.append(e2.idx)
                e2.conflicts.append(e1.idx)
        for e in self.events:
            e.conflicts = sorted(set(e.conflicts))
            logger.debug("%s has %d conflicts: %s", e, len(e.conflicts), e.conflicts)
        n = len(self.events)
        _conflict_matrix = np.full((n, n), fill_value=False, dtype=bool)
        for e1, e2 in itertools.combinations(self.events, 2):
            if e1.timeslot.overlaps(e2.timeslot):
                _conflict_matrix[e1.idx, e2.idx] = True
                _conflict_matrix[e2.idx, e1.idx] = True
        for i in range(n):
            _conflict_matrix[i, i] = True  # An event conflicts with itself
        self.mapping = {e.idx: e for e in self.events}
        self.conflict_map = {e.idx: set(e.conflicts) for e in self.events}
