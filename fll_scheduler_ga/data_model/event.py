"""Event data model for FLL scheduling."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from ..config.config import Round, RoundType, TournamentConfig
from ..config.constants import HHMM_FMT
from .location import Room, Table, get_location_type
from .time import TimeSlot

logger = logging.getLogger(__name__)

type EventMap = dict[int, Event]


@dataclass(slots=True, order=True)
class Event:
    """Data model for an event in a schedule."""

    identity: int = field(compare=True)
    roundtype: RoundType = field(compare=False)
    timeslot: TimeSlot = field(compare=False)
    location: Room | Table = field(compare=False)
    paired: "Event | None" = field(default=None, repr=False, compare=False)
    conflicts: set[int] = field(default_factory=set, repr=False, compare=False)

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
    _cached_flat_list: list[Event] = field(default=None, init=False, repr=False)
    _cached_eventmap: EventMap = field(default=None, init=False, repr=False)
    _cached_locations: dict[tuple[int, int, int], Room | Table] = field(default_factory=dict, init=False, repr=False)
    _cached_timeslots: dict[tuple[datetime, datetime], TimeSlot] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.build()
        self.flat_list()
        self.event_map()

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

    def flat_list(self) -> list[Event]:
        """Get a flat list of all Events across all RoundTypes."""
        if not self._cached_flat_list:
            self._cached_flat_list = [e for el in self._cached_events.values() for e in el]
            for i, e in enumerate(self._cached_flat_list, start=1):
                e.identity = i
        return self._cached_flat_list

    def event_map(self) -> EventMap:
        """Get a mapping of event identities to Event objects."""
        if not self._cached_eventmap:
            self._cached_eventmap = {e.identity: e for e in self.flat_list()}
        return self._cached_eventmap

    def create_events(self, r: Round) -> Iterator[Event]:
        """Generate all possible Events for a given Round configuration.

        Args:
            r (Round): The configuration of the round.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        location_type = get_location_type(r.teams_per_round)
        start = r.times[0] if r.times else r.start_time

        for i in range(r.get_num_slots()):
            if not r.times:
                stop = start + r.duration_minutes
            elif i + 1 < len(r.times):
                stop = r.times[i + 1]
            elif i + 1 == len(r.times):
                stop += r.duration_minutes

            time_cache_key = (start, stop)
            timeslot = self._cached_timeslots.setdefault(
                time_cache_key,
                TimeSlot(start, stop, start.strftime(HHMM_FMT), stop.strftime(HHMM_FMT)),
            )
            start = stop
            for j in range(1, r.num_locations + 1):
                params = {
                    "identity": j,
                    "teams_per_round": r.teams_per_round,
                }
                if hasattr(location_type, "side"):
                    cache_key1 = (j, r.teams_per_round, 1)
                    cache_key2 = (j, r.teams_per_round, 2)
                    side1_loc = self._cached_locations.setdefault(cache_key1, location_type(**params, side=1))
                    side2_loc = self._cached_locations.setdefault(cache_key2, location_type(**params, side=2))
                    event1 = Event(0, r.roundtype, timeslot, side1_loc)
                    event2 = Event(0, r.roundtype, timeslot, side2_loc)
                    event1.paired = event2
                    event2.paired = event1
                    yield event1
                    yield event2
                else:
                    cache_key = (j, r.teams_per_round)
                    location = self._cached_locations.setdefault(cache_key, location_type(**params))
                    yield Event(0, r.roundtype, timeslot, location)

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        events = self.flat_list()
        for i, event in enumerate(events):
            for other in events[i + 1 :]:
                if event.timeslot.overlaps(other.timeslot):
                    event.conflicts.add(other.identity)
                    other.conflicts.add(event.identity)
