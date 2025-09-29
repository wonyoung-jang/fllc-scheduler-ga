"""Event data model for FLL scheduling."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from .config import Round, RoundType, TournamentConfig
    from .location import Location
    from .time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    idx: int

    roundtype: RoundType
    timeslot: TimeSlot
    location: Location
    paired: Event | None = field(default=None, repr=False, compare=False)
    conflicts: set[int] = field(default_factory=set, repr=False)

    def __hash__(self) -> int:
        """Use the unique identity for hashing."""
        return self.idx

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.idx}, {self.roundtype}, {self.location}, {self.timeslot}"

    def pair(self, other: Event) -> None:
        """Pair this event with another event."""
        self.paired = other
        other.paired = self


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on Round configurations."""

    config: TournamentConfig
    _list: list[Event] = field(default_factory=list, repr=False)
    _list_singles_or_side1: list[Event] = None
    _conflict_matrix: np.ndarray = None
    _cached_timeslots: dict[tuple[datetime, datetime], TimeSlot] = field(default_factory=dict, repr=False)
    _cached_mapping: dict[int, Event] = None
    _cached_roundtypes: dict[RoundType, list[Event]] = None
    _cached_timeslots_list: dict[tuple[RoundType, TimeSlot], list[Event]] = None
    _cached_locations: dict[tuple[RoundType, Location], list[Event]] = None
    _cached_matches: dict[RoundType, list[tuple[Event, ...]]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.build()
        self.build_singles_or_side1()
        self.build_conflicts()
        self.build_conflict_matrix()
        self.as_mapping()
        self.as_timeslots()
        self.as_locations()
        self.as_matches()
        self.as_roundtypes()

        for rt, events in self._cached_roundtypes.items():
            round_events_str = f"{rt} Round has {len(events)} events."
            logger.debug("%s", round_events_str)
            for e in events:
                logger.debug("  %s | Conflicts with: %s", e, e.conflicts)

    def build(self) -> list[Event]:
        """Create and return all Events for the tournament."""
        if not self._list:
            event_idx_iter = itertools.count()
            rounds_sorted_by_start = sorted(self.config.rounds, key=lambda x: x.start_time)
            self._list.extend(e for r in rounds_sorted_by_start for e in self.create_events(r, event_idx_iter))
        return self._list

    def build_singles_or_side1(self) -> list[Event]:
        """Create and return all single-team Events or side 1 of paired Events."""
        if not self._list_singles_or_side1:
            self._list_singles_or_side1 = [
                e for e in self.build() if e.paired is None or (e.paired and e.location.side == 1)
            ]
        return self._list_singles_or_side1

    def create_events(self, r: Round, event_idx_iter: Iterator[int]) -> Iterator[Event]:
        """Generate all possible Events for a given Round configuration.

        Args:
            r (Round): The configuration of the round.
            event_idx_iter (Iterator[int]): An iterator to generate unique event IDs.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        for ts in r.timeslots:
            if r.teams_per_round == 1:
                for loc in r.locations:
                    event = Event(next(event_idx_iter), r.roundtype, ts, loc)
                    yield event
            elif r.teams_per_round == 2:
                for loc in r.locations:
                    if loc.side == 1:
                        event1 = Event(next(event_idx_iter), r.roundtype, ts, loc)
                    elif loc.side == 2:
                        event2 = Event(next(event_idx_iter), r.roundtype, ts, loc)
                        event1.pair(event2)
                        yield from (event1, event2)

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        for e1, e2 in itertools.combinations(self.build(), 2):
            if e1.timeslot.overlaps(e2.timeslot):
                e1.conflicts.add(e2.idx)
                e2.conflicts.add(e1.idx)

    def build_conflict_matrix(self) -> np.ndarray:
        """Build a conflict matrix for all events."""
        if self._conflict_matrix is None:
            n = len(self._list)
            self._conflict_matrix = np.full((n, n), fill_value=False, dtype=bool)
            for e1, e2 in itertools.combinations(self.build(), 2):
                if e1.timeslot.overlaps(e2.timeslot):
                    self._conflict_matrix[e1.idx, e2.idx] = True
                    self._conflict_matrix[e2.idx, e1.idx] = True
            for i in range(n):
                self._conflict_matrix[i, i] = True  # An event conflicts with itself
            # logger.debug("Conflict matrix:\n%s", self._conflict_matrix)
        return self._conflict_matrix

    def as_mapping(self) -> dict[int, Event]:
        """Get a mapping of event identities to Event objects."""
        if self._cached_mapping is None:
            self._cached_mapping = {e.idx: e for e in self.build()}
        return self._cached_mapping

    def as_roundtypes(self) -> dict[RoundType, list[Event]]:
        """Get a mapping of RoundTypes to their Events."""
        if self._cached_roundtypes is None:
            self._cached_roundtypes = defaultdict(list)
            for e in self.build():
                self._cached_roundtypes[e.roundtype].append(e)
        return self._cached_roundtypes

    def as_timeslots(self) -> dict[tuple[RoundType, TimeSlot], list[Event]]:
        """Get a mapping of TimeSlots to their Events."""
        if self._cached_timeslots_list is None:
            self._cached_timeslots_list = defaultdict(list)
            for e in self.build():
                self._cached_timeslots_list[(e.roundtype, e.timeslot)].append(e)
        return self._cached_timeslots_list

    def as_locations(self) -> dict[tuple[RoundType, Location], list[Event]]:
        """Get a mapping of RoundTypes to their Locations."""
        if self._cached_locations is None:
            self._cached_locations = defaultdict(list)
            for e in self.build():
                if not e.paired or (e.paired and e.location.side == 1):
                    self._cached_locations[(e.roundtype, e.location)].append(e)
        return self._cached_locations

    def as_matches(self) -> dict[RoundType, list[tuple[Event, Event]]]:
        """Get a mapping of RoundTypes to their matched Events."""
        if self._cached_matches is None:
            self._cached_matches = defaultdict(list)
            for e in self.build():
                if e.paired is None or (e.paired and e.location.side != 1):
                    continue
                self._cached_matches[e.roundtype].append((e, e.paired))
        return self._cached_matches
