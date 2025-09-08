"""Event data model for FLL scheduling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, count
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from ..config.config import Round, RoundType, TournamentConfig
    from .location import Location
    from .time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    identity: int
    roundtype: RoundType
    timeslot: TimeSlot
    location: Location
    paired: Event | None = field(default=None, repr=False, compare=False)
    conflicts: set[int] = field(default_factory=set, repr=False)

    def __hash__(self) -> int:
        """Use the unique identity for hashing."""
        return self.identity

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.identity}, {self.roundtype}, {self.location}, {self.timeslot}"

    def pair(self, other: Event) -> None:
        """Pair this event with another event."""
        self.paired = other
        other.paired = self


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on Round configurations."""

    config: TournamentConfig
    _id_counter: count = field(default_factory=count, repr=False)
    _list: list[Event] = field(default_factory=list, repr=False)
    _cached_timeslots: dict[tuple[datetime, datetime], TimeSlot] = field(default_factory=dict, repr=False)
    _cached_mapping: dict[int, Event] = field(default=None, repr=False)
    _cached_roundtypes: dict[RoundType, list[Event]] = field(default=None, repr=False)
    _cached_timeslots_list: dict[tuple[RoundType, TimeSlot], list[Event]] = field(default=None, repr=False)
    _cached_locations: dict[tuple[RoundType, Location], list[Event]] = field(default=None, repr=False)
    _cached_matches: dict[RoundType, list[tuple[Event, ...]]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        next(self._id_counter)  # Iterate to "1" for initialization

        self.build()
        self.build_conflicts()
        self.as_mapping()
        self.as_timeslots()
        self.as_locations()
        self.as_matches()
        self.as_roundtypes()

        for rt, events in self._cached_roundtypes.items():
            round_events_str = f"{rt} Round has {len(events)} events."
            logger.debug("%s", round_events_str)

    def build(self) -> list[Event]:
        """Create and return all Events for the tournament."""
        if not self._list:
            self._list.extend(
                e
                for r in sorted(
                    self.config.rounds,
                    key=lambda x: x.start_time,
                )
                for e in self.create_events(r)
            )
        return self._list

    def create_events(self, r: Round) -> Iterator[Event]:
        """Generate all possible Events for a given Round configuration.

        Args:
            r (Round): The configuration of the round.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        timeslots = r.timeslots
        teams_per_round = r.teams_per_round
        rt = r.roundtype
        locations = r.locations
        id_count = self._id_counter
        for ts in timeslots:
            if teams_per_round == 1:
                for loc in locations:
                    event = Event(next(id_count), rt, ts, loc)
                    yield event
            else:
                for loc in locations:
                    if loc.side == 1:
                        event1 = Event(next(id_count), rt, ts, loc)
                    elif loc.side == 2:
                        event2 = Event(next(id_count), rt, ts, loc)
                        event1.pair(event2)
                        yield from (event1, event2)

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        for e1, e2 in combinations(self.build(), 2):
            if e1.timeslot.overlaps(e2.timeslot):
                e1.conflicts.add(e2.identity)
                e2.conflicts.add(e1.identity)

    def as_mapping(self) -> dict[int, Event]:
        """Get a mapping of event identities to Event objects."""
        if self._cached_mapping is None:
            self._cached_mapping = {e.identity: e for e in self.build()}
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
                if not e.paired or e.location.side != 1:
                    continue
                self._cached_matches[e.roundtype].append((e, e.paired))
        return self._cached_matches
