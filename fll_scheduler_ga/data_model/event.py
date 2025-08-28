"""Event data model for FLL scheduling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from .time import TimeSlot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from ..config.config import Round, RoundType, TournamentConfig
    from .location import Location

logger = getLogger(__name__)

type EventMap = dict[int, Event]


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


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on Round configurations."""

    config: TournamentConfig
    _cached_events: dict[RoundType, list[Event]] = field(default=None, init=False, repr=False)
    _cached_list: list[Event] = field(default=None, init=False, repr=False)
    _cached_mapping: EventMap = field(default=None, init=False, repr=False)
    _cached_timeslots: dict[tuple[datetime, datetime], TimeSlot] = field(default_factory=dict, init=False, repr=False)
    _cached_timeslots_list: dict[tuple[RoundType, TimeSlot], list[Event]] = field(default=None, init=False, repr=False)
    _cached_locations: dict[tuple[RoundType, Location], list[Event]] = field(default=None, init=False, repr=False)
    _cached_matches: dict[RoundType, list[tuple[Event, ...]]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.build()
        self.as_list()
        self.as_mapping()
        self.as_timeslots()
        self.as_locations()
        self.as_matches()
        self.build_conflicts()

        for rt, events in self._cached_events.items():
            round_events_str = f"{rt} Round has {len(events)} events:"
            events_str = "\n\t  ".join(str(e) for e in events)
            logger.debug("%s\n\t  %s", round_events_str, events_str)

    def build(self) -> dict[RoundType, list[Event]]:
        """Create and return all Events for the tournament.

        Returns:
            dict[RoundType, list[Event]]: A dictionary of all Events for the tournament.

        """
        if not self._cached_events:
            self._cached_events = {
                r.roundtype: list(self.create_events(r)) for r in sorted(self.config.rounds, key=lambda x: x.start_time)
            }
        return self._cached_events

    def create_events(self, r: Round) -> Iterator[Event]:
        """Generate all possible Events for a given Round configuration.

        Args:
            r (Round): The configuration of the round.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        time_fmt = self.config.time_fmt
        locations = sorted(r.locations, key=lambda loc: (loc.identity, loc.side))
        start = r.times[0] if r.times else r.start_time

        for i in range(1, r.get_num_slots() + 1):
            if not r.times:
                stop = start + r.duration_minutes
            elif i < len(r.times):
                stop = r.times[i]
            elif i == len(r.times):
                stop += r.duration_minutes

            time_key = (start, stop)
            timeslot = self._cached_timeslots.setdefault(time_key, TimeSlot(start, stop, time_fmt))
            start = stop

            if r.teams_per_round == 1:
                for loc in locations:
                    event = Event(0, r.roundtype, timeslot, loc)
                    yield event
            else:
                for loc in locations:
                    if loc.side == 1:
                        event1 = Event(0, r.roundtype, timeslot, loc)
                    elif loc.side == 2:
                        event2 = Event(0, r.roundtype, timeslot, loc)
                        event1.paired = event2
                        event2.paired = event1
                        yield from (event1, event2)

    def as_list(self) -> list[Event]:
        """Get a flat list of all Events across all RoundTypes."""
        if self._cached_list is None:
            self._cached_list = [e for el in self._cached_events.values() for e in el]
            for i, event in enumerate(self._cached_list, start=1):
                event.identity = i
        return self._cached_list

    def as_mapping(self) -> EventMap:
        """Get a mapping of event identities to Event objects."""
        if self._cached_mapping is None:
            as_list = self.as_list()
            self._cached_mapping = {e.identity: e for e in as_list}
        return self._cached_mapping

    def as_timeslots(self) -> dict[tuple[RoundType, TimeSlot], list[Event]]:
        """Get a mapping of TimeSlots to their Events."""
        if self._cached_timeslots_list is None:
            self._cached_timeslots_list = defaultdict(list)
            for event in self.as_list():
                self._cached_timeslots_list[(event.roundtype, event.timeslot)].append(event)
        return self._cached_timeslots_list

    def as_locations(self) -> dict[tuple[RoundType, Location], list[Event]]:
        """Get a mapping of RoundTypes to their Locations."""
        if self._cached_locations is None:
            self._cached_locations = defaultdict(list)
            for event in self.as_list():
                self._cached_locations[(event.roundtype, event.location)].append(event)
        return self._cached_locations

    def as_matches(self) -> dict[RoundType, list[tuple[Event, ...]]]:
        """Get a mapping of RoundTypes to their matched Events."""
        if self._cached_matches is None:
            self._cached_matches = defaultdict(list)
            for event in self.as_list():
                if not event.paired or event.location.side != 1:
                    continue
                self._cached_matches[event.roundtype].append((event, event.paired))
        return self._cached_matches

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        events = self.as_list()
        for i, event in enumerate(events, start=1):
            for other in events[i:]:
                if event.timeslot.overlaps(other.timeslot):
                    event.conflicts.add(other.identity)
                    other.conflicts.add(event.identity)
