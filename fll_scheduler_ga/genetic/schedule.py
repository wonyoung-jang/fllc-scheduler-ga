"""Represents a schedule (individual) with its associated fitness score."""

from collections import defaultdict
from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass, field

from ..config.config import RoundType
from ..data_model.event import Event
from ..data_model.team import Team, TeamMap

type Population = list[Schedule]
type Individual = dict[Event, int]
type Match = tuple[Event, Event, Team, Team]


@dataclass(slots=True, order=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    _teams: TeamMap = field(default_factory=dict, compare=False)
    _schedule: Individual = field(default_factory=dict, compare=False)
    fitness: tuple[float, ...] | None = field(default=None, compare=False)
    rank: int = field(default=9999, compare=True)
    crowding: float = field(default=0.0, compare=False)
    _cached_all_teams: list[Team] = field(default=None, init=False, repr=False, compare=False)
    _cached_matches: dict[RoundType, list[Match]] = field(default=None, init=False, repr=False, compare=False)
    _hash: int = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return len(self._schedule)

    def __getitem__(self, event: Event) -> Team:
        """Get the team assigned to a specific event."""
        try:
            team_id = self._schedule[event]
            return self._teams[team_id]
        except KeyError:
            msg = f"The event {event} is not scheduled."
            raise KeyError(msg) from None

    def __setitem__(self, event: Event, team: Team) -> None:
        """Assign a team to a specific event."""
        self._schedule[event] = team.identity
        self._cached_matches = None
        self._cached_all_teams = None
        self._hash = None

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return event in self._schedule

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return self._schedule == other._schedule

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._hash is None:
            canonical_representation = tuple(
                sorted((event.identity, team_id) for event, team_id in self._schedule.items())
            )
            self._hash = hash(canonical_representation)
        return self._hash

    def get_matches(self) -> dict[RoundType, list[Match]]:
        """Get all matches in the schedule."""
        if self._cached_matches is not None:
            return self._cached_matches

        self._cached_matches = defaultdict(list)
        for event1, t1 in self._schedule.items():
            if not (event2 := event1.paired_event) or event1.location.side != 1:
                continue

            rt = event1.round_type
            if t2 := self._schedule[event2]:
                team1 = self._teams[t1]
                team2 = self._teams[t2]
                self._cached_matches[rt].append((event1, event2, team1, team2))

        return self._cached_matches

    # def clone(self) -> "Schedule":
    #     """Create a deep copy of the Schedule instance."""
    #     new_teams = {identity: team.clone() for identity, team in self._teams.items()}
    #     new_individual = {event: new_teams[team_id] for event, team_id in self._schedule.items()}
    #     return Schedule(new_teams, new_individual, self.fitness, self.rank, self.crowding)

    def all_teams(self) -> list[Team]:
        """Return a list of all teams in the schedule."""
        if self._cached_all_teams is None:
            self._cached_all_teams = list(self._teams.values())
        return self._cached_all_teams

    def keys(self) -> KeysView[Event]:
        """Return an iterator over the events (keys)."""
        return self._schedule.keys()

    def values(self) -> ValuesView[Team]:
        """Return an iterator over the assigned teams/matches (values)."""
        return self._schedule.values()

    def items(self) -> ItemsView[Event, Team]:
        """Return an iterator over the (event, team/match) pairs."""
        return self._schedule.items()

    def get_team(self, team_id: int) -> Team:
        """Get a team object by its identity."""
        return self._teams[team_id]
