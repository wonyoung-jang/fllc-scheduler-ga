"""Represents a schedule (individual) with its associated fitness score."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .event import Event, EventProperties

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    schedule: np.ndarray = None
    fitness: np.ndarray = None
    team_fitnesses: np.ndarray = None
    rank: int = -1

    origin: str = "Builder"
    mutations: int = 0
    clones: int = 0

    _hash: int = None

    team_events: dict[int, set[int]] = None
    team_rounds: np.ndarray = None

    # Class variables
    teams: ClassVar[np.ndarray]
    event_map: ClassVar[dict[int, Event]]
    event_properties: ClassVar[EventProperties]
    team_roundreqs_array: ClassVar[np.ndarray]
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
            self.team_rounds = Schedule.team_roundreqs_array.copy()

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return np.count_nonzero(self.schedule >= 0)

    def __getitem__(self, event: int) -> int | None:
        """Get the team assigned to a specific event."""
        return self.schedule[event]

    def __setitem__(self, event: int, team: int) -> None:
        """Assign a team to a specific event."""
        self.schedule[event] = team
        self._hash = None

    def __delitem__(self, event: int) -> None:
        """Remove a specific event from the schedule."""
        self.schedule[event] = -1
        self._hash = None

    def __contains__(self, event: int) -> bool:
        """Check if a specific event is scheduled."""
        return self.schedule[event] != -1

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._hash is None:
            self._hash = id(self)
        return self._hash

    def swap_assignment(self, team: int, old_event: int, new_event: int) -> None:
        """Switch an event for a team in the schedule."""
        self.unassign(team, old_event)
        self.assign(team, new_event)

    def assign(self, team: int, event: int) -> None:
        """Add an event to a team's scheduled events."""
        roundtype = Schedule.event_properties.roundtype_idx[event]
        self.team_events[team].add(event)
        self.team_rounds[team, roundtype] -= 1
        self[event] = team

    def unassign(self, team: int, event: int) -> None:
        """Remove an event from a team's scheduled events."""
        roundtype = Schedule.event_properties.roundtype_idx[event]
        self.team_events[team].remove(event)
        self.team_rounds[team, roundtype] += 1
        del self[event]

    def needs_round(self, team: int, roundtype: int) -> bool:
        """Check if a team still needs to participate in a given round type."""
        return self.team_rounds[team, roundtype] > 0

    def all_rounds_needed(self, roundtype: int) -> np.ndarray:
        """Return all teams that still need roundtype."""
        return np.where(self.team_rounds[:, roundtype] > 0)[0]

    def any_rounds_needed(self) -> bool:
        """Check if any team still needs rounds."""
        return np.any(self.team_rounds.sum(axis=1) > 0)

    def conflicts(self, team: int, new_event: int, *, ignore: int | None = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            team (int): The team to check for conflicts.
            new_event (int): The new event to check for conflicts.
            ignore (int): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        team_events = self.team_events[team]
        events_to_check = team_events

        if ignore is not None and ignore in team_events:
            events_to_check = events_to_check - {ignore}

        if new_event in events_to_check:
            return True

        new_conflicts = Schedule.event_map[new_event].conflicts
        return not events_to_check.isdisjoint(new_conflicts)

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        return Schedule(
            schedule=self.schedule.copy(),
            fitness=self.fitness.copy() if self.fitness is not None else None,
            team_fitnesses=self.team_fitnesses.copy() if self.team_fitnesses is not None else None,
            rank=self.rank,
            origin=self.origin,
            mutations=self.mutations,
            clones=self.clones + 1,
            _hash=self._hash,
            team_events={k: v.copy() for k, v in self.team_events.items()},
            team_rounds=self.team_rounds.copy(),
        )

    def scheduled_events(self) -> np.ndarray:
        """Return the indices of scheduled events."""
        return np.nonzero(self.schedule >= 0)[0]

    def unscheduled_events(self) -> np.ndarray:
        """Return the indices of unscheduled events."""
        return np.nonzero(self.schedule == -1)[0]

    def normalized_teams(self) -> dict[int, int | str]:
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
