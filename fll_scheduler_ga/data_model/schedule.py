"""Represents a schedule (individual) with its associated fitness score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from .event import Event
from .team import Team

if TYPE_CHECKING:
    from collections.abc import ItemsView, KeysView, ValuesView


type Population = list[Schedule]
type Individual = dict[Event, int]
type Match = tuple[Event, Event, Team, Team]


@dataclass(slots=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    teams: dict[int, Team] = field(default_factory=dict)
    schedule: Individual = field(default_factory=dict)
    fitness: tuple[float, ...] | None = field(default=None)
    rank: int = field(default=99)

    ref_point: int = field(default=None, repr=False)
    ref_distance: float = field(default=None, repr=False)

    _cached_all_teams: list[Team] = field(default=None, repr=False)
    _cached_normalized_teams: dict[int, int] = field(default=None, repr=False)
    _cached_hash: int = field(default=None, repr=False)
    _cached_canonical_representation: tuple[tuple[int, ...], ...] = field(default=None, repr=False)

    team_identities: ClassVar[dict[int, int | str]]

    def __post_init__(self) -> None:
        """Post-initialization processing."""

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return len(self.schedule)

    def __getitem__(self, event: Event) -> Team | None:
        """Get the team assigned to a specific event."""
        team_id = self.schedule.get(event, None)
        return self.teams.get(team_id, None)

    def __setitem__(self, event: Event, team: Team) -> None:
        """Assign a team to a specific event."""
        self.schedule[event] = team.identity

    def __delitem__(self, event: Event) -> None:
        """Remove a specific event from the schedule."""
        del self.schedule[event]

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return event in self.schedule

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return self.canonical_representation() == other.canonical_representation()

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._cached_hash is None:
            self._cached_hash = hash(self.canonical_representation())
        return self._cached_hash

    @classmethod
    def set_team_identities(cls, identities: dict[int, int | str]) -> None:
        """Set the team identities for the schedule."""
        cls.team_identities = identities

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        clone = Schedule(
            teams={i: t.clone() for i, t in self.teams.items()},
            schedule=dict(self.schedule.items()),
        )
        clone.clear_cache()
        return clone

    def clear_cache(self) -> None:
        """Clear cached values to ensure fresh calculations."""
        self._cached_all_teams = None
        self._cached_normalized_teams = None
        self._cached_hash = None
        self._cached_canonical_representation = None

    def keys(self) -> KeysView[Event]:
        """Return an iterator over the events (keys)."""
        return self.schedule.keys()

    def values(self) -> ValuesView[Team]:
        """Return an iterator over the assigned teams/matches (values)."""
        return self.schedule.values()

    def items(self) -> ItemsView[Event, Team]:
        """Return an iterator over the (event, team/match) pairs."""
        return self.schedule.items()

    def assign_single(self, event: Event, team: Team) -> None:
        """Assign a single-team event to a team."""
        team.add_event(event)
        self[event] = team

    def assign_match(self, event1: Event, event2: Event, team1: Team, team2: Team) -> None:
        """Assign a match event to two teams."""
        team1.add_event(event1)
        team2.add_event(event2)
        team1.add_opponent(team2)
        team2.add_opponent(team1)
        self[event1] = team1
        self[event2] = team2

    def destroy_event(self, e1: Event) -> None:
        """Destroy an event and its pair if it exists."""
        if (e2 := e1.paired) and e1.location.side == 1:
            t1 = self[e1]
            t2 = self[e2]
            if t1 and t2:
                t1.remove_event(e1)
                t2.remove_event(e2)
                t1.remove_opponent(t2)
                t2.remove_opponent(t1)
                del self[e1]
                del self[e2]
        elif not e2:
            t1 = self[e1]
            if t1:
                t1.remove_event(e1)
                del self[e1]

    def get_team(self, team_id: int | Team) -> Team | None:
        """Get a team object by its identity."""
        if isinstance(team_id, Team):
            team_id = team_id.identity
        return self.teams.get(team_id)

    def all_teams(self) -> list[Team]:
        """Return a list of all teams in the schedule."""
        if self._cached_all_teams is None:
            self._cached_all_teams = list(self.teams.values())
        return self._cached_all_teams

    def canonical_representation(self) -> frozenset[frozenset[int]]:
        """Get a canonical representation of the schedule for hashing."""
        if self._cached_canonical_representation is None:
            team_events = (frozenset(t.events) for t in self.teams.values())
            self._cached_canonical_representation = frozenset(team_events)
        return self._cached_canonical_representation

    def normalized_teams(self) -> dict[int, int]:
        """Normalize the schedule by reassigning team identities."""
        if self._cached_normalized_teams is None:
            self._cached_normalized_teams = {}
            count = 1
            for _, team in sorted(self.items(), key=lambda i: (i[0].identity)):
                if team in self._cached_normalized_teams:
                    continue

                self._cached_normalized_teams[team] = self.team_identities.get(count, count)
                if count == len(self.teams):
                    break
                count += 1
        return self._cached_normalized_teams
