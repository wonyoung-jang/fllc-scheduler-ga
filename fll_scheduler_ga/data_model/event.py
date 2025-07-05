"""Event data model for FLL scheduling."""

import itertools
import logging
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..config.config import Round, RoundType, TournamentConfig
from .location import Room, Table, get_location_type
from .time import HHMM_FMT, TimeSlot

logger = logging.getLogger(__name__)


@dataclass(slots=True, unsafe_hash=True)
class Event:
    """Data model for an event in a schedule."""

    identity: int = field(hash=True)
    round_type: RoundType = field(compare=False)
    time_slot: TimeSlot = field(compare=False)
    location: Room | Table = field(compare=False)
    paired_event: "Event | None" = field(default=None, repr=False, compare=False)

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"Round type: {self.round_type}, {self.location}, {self.time_slot}"


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on Round configurations."""

    config: TournamentConfig
    _id_counter: itertools.count = field(default_factory=itertools.count, init=False)
    _cached_events: dict[RoundType, set[Event]] = field(default_factory=dict, init=False, repr=False)

    def build(self) -> dict[RoundType, set[Event]]:
        """Create and return all Events for the tournament.

        Returns:
            dict[RoundType, set[Event]]: A dictionary of all Events for the tournament.

        """
        if not self._cached_events:
            self._cached_events = {r.round_type: set(self.create_events(r)) for r in self.config.rounds}
        return self._cached_events

    def create_events(self, r: Round) -> Generator[Event]:
        """Generate all possible Events for a given Round configuration.

        Args:
            r (Round): The configuration of the round.

        Yields:
            Event: An event for the round with a time slot and a location.

        """
        start = datetime.strptime(r.start_time, HHMM_FMT).replace(tzinfo=UTC)
        location_type = get_location_type(r.teams_per_round)

        for _ in range(r.num_slots):
            stop = start + r.duration_minutes
            time_slot = TimeSlot(start, stop, start.strftime(HHMM_FMT), stop.strftime(HHMM_FMT))
            start = stop

            for i in range(1, r.num_locations + 1):
                params = {"identity": i, "teams_per_round": r.teams_per_round}
                if hasattr(location_type, "side"):
                    side1_loc = location_type(**params, side=1)
                    side2_loc = location_type(**params, side=2)
                    event1 = Event(next(self._id_counter), r.round_type, time_slot, side1_loc)
                    event2 = Event(next(self._id_counter), r.round_type, time_slot, side2_loc)
                    event1.paired_event = event2
                    event2.paired_event = event1
                    yield event1
                    yield event2
                else:
                    location = location_type(**params)
                    yield Event(next(self._id_counter), r.round_type, time_slot, location)


@dataclass(slots=True)
class EventMap:
    """Mapping of event identities to Event instances."""

    event_factory: EventFactory
    events: list[Event] = field(default_factory=list, init=False, repr=False)
    conflicts: dict[int, set[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to create the event availability map."""
        self.events = [event for el in self.event_factory.build().values() for event in el]
        self.conflicts = defaultdict(set)
        self._compile_map()

    def _compile_map(self) -> None:
        """Compile the event availability map."""
        for curr_event in self.events:
            self.conflicts[curr_event.identity] = {
                e.identity
                for e in self.events
                if e.identity != curr_event.identity and curr_event.time_slot.overlaps(e.time_slot)
            }

        for k, v in sorted(self.conflicts.items(), key=lambda item: item[0]):
            logger.debug("Event %d conflicts with %d other events: %s", k, len(v), v)
