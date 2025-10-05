"""Represents a schedule (individual) with its associated fitness score."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .event import Event
from .team import Team

if TYPE_CHECKING:
    from .config import RoundType

type Population = list[Schedule]
type Individual = dict[Event, int]
type Match = tuple[Event, Event, Team, Team]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    teams: list[Team] = None
    schedule: np.ndarray[int] = None
    fitness: np.ndarray[float] = None
    team_fitnesses: np.ndarray[float] = None
    rank: int = -1

    origin: str = "Builder"
    mutations: int = 0
    clones: int = 0

    _hash: int = None

    team_events: dict[int, set[int]] = None
    team_rounds: dict[int, dict[RoundType, int]] = None

    team_roundreqs: ClassVar[dict[int, dict[RoundType, int]]]
    team_identities: ClassVar[dict[int, int | str]]
    total_num_events: ClassVar[int]
    conflict_matrix: ClassVar[np.ndarray]

    def __post_init__(self) -> None:
        """Post-initialization to set up fitness array."""
        if self.schedule is None:
            self.schedule = np.full(Schedule.total_num_events, -1, dtype=int)

        if self.team_events is None:
            self.team_events = defaultdict(set)

        if self.team_rounds is None:
            self.team_rounds = {team.idx: Schedule.team_roundreqs.copy() for team in self.teams}

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return self.schedule.size - np.count_nonzero(self.schedule == -1)

    def __getitem__(self, event: Event) -> Team | None:
        """Get the team assigned to a specific event."""
        if (team_id := self.schedule[event.idx]) == -1:
            return None
        return self.teams[team_id]

    def __setitem__(self, event: Event, team: Team) -> None:
        """Assign a team to a specific event."""
        self.schedule[event.idx] = team.idx
        self._hash = None

    def __delitem__(self, event: Event) -> None:
        """Remove a specific event from the schedule."""
        self.schedule[event.idx] = -1
        self._hash = None

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return self.schedule[event.idx] != -1

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._hash is None:
            self._hash = hash(frozenset(frozenset(events) for events in self.team_events.values()))
        return self._hash

    @classmethod
    def set_team_identities(cls, identities: dict[int, int | str]) -> None:
        """Set the team identities for the schedule."""
        cls.team_identities = identities

    @classmethod
    def set_total_num_events(cls, total: int) -> None:
        """Set the total number of events for the schedule."""
        cls.total_num_events = total

    @classmethod
    def set_team_roundreqs(cls, roundreqs: dict[int, dict[str, int]]) -> None:
        """Set the team round requirements for the schedule."""
        cls.team_roundreqs = roundreqs

    @classmethod
    def set_conflict_matrix(cls, matrix: np.ndarray) -> None:
        """Set the conflict matrix for the schedule."""
        cls.conflict_matrix = matrix

    def swap_assignment(self, team: Team, old_event: Event, new_event: Event) -> None:
        """Switch an event for a team in the schedule."""
        self.unassign(team, old_event)
        self.assign(team, new_event)

    def assign(self, team: Team, event: Event) -> None:
        """Add an event to a team's scheduled events."""
        ti = team.idx
        ei = event.idx
        self.team_events[ti].add(ei)
        self.team_rounds[ti][event.roundtype] -= 1
        self.schedule[ei] = ti
        self._hash = None

    def unassign(self, team: Team, event: Event) -> None:
        """Remove an event from a team's scheduled events."""
        ti = team.idx
        ei = event.idx
        self.team_events[ti].remove(ei)
        self.team_rounds[ti][event.roundtype] += 1
        self.schedule[ei] = -1
        self._hash = None

    def team_rounds_needed(self, team: Team) -> bool:
        """Check if any rounds still needed for the team."""
        return sum(self.team_rounds[team.idx].values()) > 0

    def needs_round(self, team: Team, roundtype: RoundType) -> bool:
        """Check if a team still needs to participate in a given round type."""
        return self.team_rounds[team.idx][roundtype] > 0

    def conflicts(self, team: Team, new_event: Event, *, ignore: Event = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            team (Team): The team to check for conflicts.
            new_event (Event): The new event to check for conflicts.
            ignore (Event): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        team_events = self.team_events[team.idx]
        events_to_check = team_events

        if ignore and ignore.idx in team_events:
            events_to_check = team_events - {ignore.idx}

        conflict_found = new_event.idx in events_to_check

        if conflict_found:
            return True

        new_conflicts = set(new_event.conflicts)
        return bool(not conflict_found and new_conflicts and not events_to_check.isdisjoint(new_conflicts))

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        return Schedule(
            teams=self.teams.copy(),
            schedule=self.schedule.copy(),
            fitness=self.fitness.copy() if self.fitness is not None else None,
            team_fitnesses=self.team_fitnesses.copy() if self.team_fitnesses is not None else None,
            rank=self.rank,
            origin=self.origin,
            mutations=self.mutations,
            clones=self.clones + 1,
            _hash=self._hash,
            team_events={k: v.copy() for k, v in self.team_events.items()},
            team_rounds={k: v.copy() for k, v in self.team_rounds.items()},
        )

    def scheduled_events(self) -> np.ndarray:
        """Return the indices of scheduled events."""
        return np.where(self.schedule >= 0)[0]

    def unscheduled_events(self) -> np.ndarray:
        """Return the indices of unscheduled events."""
        return np.where(self.schedule == -1)[0]

    def destroy_event(self, event1: Event) -> None:
        """Destroy an event, whether single or match."""
        if event2 := event1.paired:
            if (team1 := self[event1]) and (team2 := self[event2]):
                self.unassign(team1, event1)
                self.unassign(team2, event2)
        elif team := self[event1]:
            self.unassign(team, event1)

    def get_team(self, team_id: int) -> Team:
        """Get a team object by its identity."""
        return self.teams[team_id]

    def normalized_teams(self) -> dict[int, int]:
        """Normalize the schedule by reassigning team identities."""
        normalized_teams = {}
        count = 1
        for team in self.schedule:
            if team in normalized_teams:
                continue

            normalized_teams[team] = self.team_identities.get(count, count)
            if count == len(self.teams):
                break
            count += 1
        return normalized_teams
