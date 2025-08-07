"""Event data model for FLL scheduling."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from ..config.config import Round, RoundType, TournamentConfig
from ..config.constants import HHMM_FMT
from .location import Location
from .time import TimeSlot

logger = logging.getLogger(__name__)

type EventMap = dict[int, Event]


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    identity: int
    roundtype: RoundType
    timeslot: TimeSlot
    location: Location
    paired: "Event | None" = field(default=None, repr=False, compare=False)
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
    _cached_locations: dict[tuple[int, int, int], Location] = field(default_factory=dict, init=False, repr=False)
    _cached_timeslots: dict[tuple[datetime, datetime], TimeSlot] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.build()
        self.as_list()
        self.as_mapping()
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
        locations = sorted(r.locations, key=lambda loc: (loc.identity, loc.side))
        start = r.times[0] if r.times else r.start_time

        for i in range(1, r.get_num_slots() + 1):
            if not r.times:
                stop = start + r.duration_minutes
            elif i < len(r.times):
                stop = r.times[i]
            elif i == len(r.times):
                stop += r.duration_minutes

            time_cache_key = (start, stop)
            timeslot = self._cached_timeslots.setdefault(
                time_cache_key,
                TimeSlot(start, stop, start.strftime(HHMM_FMT), stop.strftime(HHMM_FMT)),
            )
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
                        yield event1
                        yield event2

    def as_list(self) -> list[Event]:
        """Get a flat list of all Events across all RoundTypes."""
        if not self._cached_list:
            self._cached_list = [e for el in self._cached_events.values() for e in el]
            for i, event in enumerate(self._cached_list, start=1):
                event.identity = i
        return self._cached_list

    def as_mapping(self) -> EventMap:
        """Get a mapping of event identities to Event objects."""
        if not self._cached_mapping:
            self._cached_mapping = {e.identity: e for e in self.as_list()}
        return self._cached_mapping

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        events = self.as_list()
        for i, event in enumerate(events):
            for other in events[i + 1 :]:
                if event.timeslot.overlaps(other.timeslot):
                    event.conflicts.add(other.identity)
                    other.conflicts.add(event.identity)
