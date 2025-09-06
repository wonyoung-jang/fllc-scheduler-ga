"""Team data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.config import RoundType, TournamentConfig
    from .event import Event
    from .time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class Team:
    """Data model for a team in the FLL Scheduler GA."""

    identity: int
    roundreqs: dict[RoundType, int]
    fitness: tuple[float, ...] = field(default=None)
    events: set[int] = field(default_factory=set, repr=False)
    timeslots: list[TimeSlot] = field(default_factory=list, repr=False)
    opponents: list[int] = field(default_factory=list, repr=False)
    tables: list[int] = field(default_factory=list, repr=False)

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return self.identity

    def clone(self) -> Team:
        """Create a deep copy of the team."""
        return Team(
            identity=self.identity,
            roundreqs=self.roundreqs.copy(),
            fitness=self.fitness,
            events=self.events.copy(),
            timeslots=self.timeslots[:],
            opponents=self.opponents[:],
            tables=self.tables[:],
        )

    def rounds_needed(self) -> bool:
        """Get the total number of rounds still needed for the team."""
        return sum(self.roundreqs.values()) > 0

    def needs_round(self, round_type: RoundType) -> bool:
        """Check if the team still needs to participate in a given round type."""
        return self.roundreqs[round_type] > 0

    def switch_event(self, old_event: Event, new_event: Event) -> None:
        """Switch an event for the team."""
        self.remove_event(old_event)
        self.add_event(new_event)

    def remove_event(self, event: Event) -> None:
        """Unbook a team from an event."""
        self.roundreqs[event.roundtype] += 1
        self.events.remove(event.identity)
        self.timeslots.remove(event.timeslot)
        if event.paired:
            self.tables.remove(event.location)

    def add_event(self, event: Event) -> None:
        """Book a team for an event."""
        self.roundreqs[event.roundtype] -= 1
        self.events.add(event.identity)
        self.timeslots.append(event.timeslot)
        if event.paired:
            self.tables.append(event.location)

    def switch_opponent(self, old_opponent: Team, new_opponent: Team) -> None:
        """Switch the opponent for a given event."""
        self.remove_opponent(old_opponent)
        self.add_opponent(new_opponent)

    def remove_opponent(self, opponent: Team) -> None:
        """Remove an opponent for a given event."""
        self.opponents.remove(opponent.identity)

    def add_opponent(self, opponent: Team) -> None:
        """Add an opponent for a given event."""
        self.opponents.append(opponent.identity)

    def conflicts(self, new_event: Event, *, ignore: Event = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            new_event (Event): The new event to check for conflicts.
            ignore (Event): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        _events = self.events

        if ignore and ignore.identity in _events:
            _events.remove(ignore.identity)

        conflict_found = new_event.identity in _events

        if not conflict_found and new_event.conflicts and not _events.isdisjoint(new_event.conflicts):
            conflict_found = True

        if ignore:
            _events.add(ignore.identity)

        return conflict_found

    def get_fitness_keys(self) -> Iterator[frozenset[int], int, int]:
        """Get all keys used for fitness calculation."""
        yield self.break_time_key()
        yield self.table_consistency_key()
        yield self.opponent_variety_key()

    def break_time_key(self) -> frozenset[int]:
        """Get a key for the break time cache based on the team's events."""
        return frozenset(self.timeslots)

    def table_consistency_key(self) -> int:
        """Get a key for the table consistency cache based on the team's events."""
        return len(set(self.tables))

    def opponent_variety_key(self) -> int:
        """Get a key for the opponent variety cache based on the team's events."""
        return len(set(self.opponents))


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig

    def build(self) -> dict[int, Team]:
        """Create a mapping of team identities to Team instances.

        Returns:
            dict[int, Team]: A mapping of team identities to Team instances.

        """
        return {
            i: Team(
                identity=i,
                roundreqs=self.config.round_requirements.copy(),
            )
            for i in range(1, self.config.num_teams + 1)
        }
