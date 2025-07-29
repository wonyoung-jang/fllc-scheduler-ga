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
    round_type: RoundType = field(compare=False)
    timeslot: TimeSlot = field(compare=False)
    location: Room | Table = field(compare=False)
    paired_event: "Event | None" = field(default=None, repr=False, compare=False)

    def __hash__(self) -> int:
        """Use the unique identity for hashing."""
        return self.identity

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.identity}, {self.round_type}, {self.location}, {self.timeslot}"


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

    def build(self) -> dict[RoundType, list[Event]]:
        """Create and return all Events for the tournament.

        Returns:
            dict[RoundType, list[Event]]: A dictionary of all Events for the tournament.

        """
        if not self._cached_events:
            self._cached_events = {
                r.round_type: list(self.create_events(r))
                for r in sorted(self.config.rounds, key=lambda x: x.start_time)
            }
        return self._cached_events

    def flat_list(self) -> list[Event]:
        """Get a flat list of all Events across all RoundTypes."""
        if not self._cached_flat_list:
            self._cached_flat_list = [e for el in self._cached_events.values() for e in el]
            for i, e in enumerate(self._cached_flat_list):
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
        start = r.start_time
        times = r.times
        teams_per_round = r.teams_per_round
        round_type = r.round_type
        num_slots = r.get_num_slots()
        num_locations = r.num_locations
        duration_minutes = r.duration_minutes
        location_type = get_location_type(teams_per_round)

        if times:
            start = times[0]

        for i in range(num_slots):
            if not times:
                stop = start + duration_minutes
            elif i + 1 < len(times):
                stop = times[i + 1]
            elif i + 1 == len(times):
                stop += duration_minutes

            time_cache_key = (start, stop)
            timeslot = self._cached_timeslots.setdefault(
                time_cache_key,
                TimeSlot(start, stop, start.strftime(HHMM_FMT), stop.strftime(HHMM_FMT)),
            )
            start = stop
            for j in range(1, num_locations + 1):
                params = {
                    "identity": j,
                    "teams_per_round": teams_per_round,
                }
                if hasattr(location_type, "side"):
                    cache_key1 = (j, teams_per_round, 1)
                    cache_key2 = (j, teams_per_round, 2)
                    side1_loc = self._cached_locations.setdefault(cache_key1, location_type(**params, side=1))
                    side2_loc = self._cached_locations.setdefault(cache_key2, location_type(**params, side=2))
                    event1 = Event(0, round_type, timeslot, side1_loc)
                    event2 = Event(0, round_type, timeslot, side2_loc)
                    event1.paired_event = event2
                    event2.paired_event = event1
                    yield event1
                    yield event2
                else:
                    cache_key = (j, teams_per_round)
                    location = self._cached_locations.setdefault(cache_key, location_type(**params))
                    yield Event(0, round_type, timeslot, location)


@dataclass(slots=True)
class EventConflicts:
    """Mapping of event identities to their conflicting events."""

    event_factory: EventFactory
    conflicts: dict[int, set[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to create the event availability map."""
        events = self.event_factory.flat_list()
        self.conflicts = {e.identity: set() for e in events}

        for event in events:
            logger.debug("Event %d: %s", event.identity, event)
            for other in events:
                if other.identity == event.identity:
                    continue

                if event.timeslot.overlaps(other.timeslot):
                    self.conflicts[event.identity].add(other.identity)

        for k, v in self.conflicts.items():
            logger.debug("Event %d conflicts with %d other events: %s", k, len(v), v)
