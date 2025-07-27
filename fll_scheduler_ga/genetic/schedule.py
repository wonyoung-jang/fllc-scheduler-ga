"""Represents a schedule (individual) with its associated fitness score."""

from collections import defaultdict
from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.event import Event
from ..data_model.team import Team, TeamMap

if TYPE_CHECKING:
    from ..config.config import RoundType

type Population = list[Schedule]
type Individual = dict[Event, int]
type Match = tuple[Event, Event, Team, Team]


@dataclass(slots=True, order=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    teams: TeamMap = field(default_factory=dict, compare=False)
    schedule: Individual = field(default_factory=dict, compare=False)
    fitness: tuple[float, ...] | None = field(default=(0, 0, 0), compare=False)
    rank: int = field(default=10, compare=True)

    normalized_fitness: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    ref_point_idx: int = field(default=None, init=False, repr=False, compare=False)
    distance_to_ref_point: float = field(default=None, init=False, repr=False, compare=False)

    _cached_all_teams: list[Team] = field(default=None, init=False, repr=False, compare=False)
    _cached_normalized_teams: dict[int, int] = field(default=None, init=False, repr=False, compare=False)
    _cached_matches: dict["RoundType", list[Match]] = field(default=None, init=False, repr=False, compare=False)
    _cached_hash: int = field(default=None, init=False, repr=False)
    _cached_canonical_representation: tuple[tuple[int, ...], ...] = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return len(self.schedule)

    def __getitem__(self, event: Event) -> Team:
        """Get the team assigned to a specific event."""
        try:
            team_id = self.schedule[event]
            return self.teams[team_id]
        except KeyError:
            msg = f"The event {event} is not scheduled."
            raise KeyError(msg) from None

    def __setitem__(self, event: Event, team: Team) -> None:
        """Assign a team to a specific event."""
        self.schedule[event] = team.identity
        self._cached_all_teams = None
        self._cached_normalized_teams = None
        self._cached_matches = None
        self._cached_hash = None
        self._cached_canonical_representation = None

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return event in self.schedule

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return self._get_canonical_representation() == other._get_canonical_representation()

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._cached_hash is None:
            self._cached_hash = hash(self._get_canonical_representation())
        return self._cached_hash

    def _get_canonical_representation(self) -> tuple[tuple[int, ...], ...]:
        """Get a canonical representation of the schedule for hashing."""
        if self._cached_canonical_representation is not None:
            return self._cached_canonical_representation

        if not self.schedule:
            return ()

        team_to_event_map = defaultdict(list)
        for event, team_id in self.schedule.items():
            team_to_event_map[team_id].append(event.identity)

        sorted_events = [tuple(sorted(events)) for events in team_to_event_map.values()]
        sorted_events.sort()
        self._cached_canonical_representation = tuple(sorted_events)
        return self._cached_canonical_representation

    def get_matches(self) -> dict["RoundType", list[Match]]:
        """Get all matches in the schedule."""
        if self._cached_matches is not None:
            return self._cached_matches

        self._cached_matches = defaultdict(list)
        for event1, t1 in self.schedule.items():
            if not (event2 := event1.paired_event) or event1.location.side != 1:
                continue

            rt = event1.round_type
            if t2 := self.schedule[event2]:
                self._cached_matches[rt].append((event1, event2, self.teams[t1], self.teams[t2]))

        return self._cached_matches

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

    def all_teams(self) -> list[Team]:
        """Return a list of all teams in the schedule."""
        if self._cached_all_teams is None:
            self._cached_all_teams = list(self.teams.values())
        return self._cached_all_teams

    def get_team(self, team_id: int) -> Team:
        """Get a team object by its identity."""
        return self.teams[team_id]

    def normalize_teams(self) -> dict[int, int]:
        """Normalize the schedule by reassigning team identities."""
        if self._cached_normalized_teams is None:
            len_teams = len(self.teams)
            self._cached_normalized_teams = {}
            for count, (_, team) in enumerate(sorted(self.items(), key=lambda i: (i[0].identity)), start=1):
                self._cached_normalized_teams[team] = count
                if count == len_teams:
                    break
        return self._cached_normalized_teams
