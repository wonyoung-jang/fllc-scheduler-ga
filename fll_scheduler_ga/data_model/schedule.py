"""Represents a schedule (individual) with its associated fitness score."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .event import EventProperties

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleContext:
    """Holds class-level context for Schedule instances."""

    conflict_map: dict[int, set[int]]
    event_props: EventProperties
    teams_list: np.ndarray
    teams_roundreqs_arr: np.ndarray
    n_total_events: int


@dataclass(slots=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    schedule: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness: np.ndarray = field(default_factory=lambda: np.array([]))
    team_fitnesses: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = -1
    origin: str = "Builder"
    mutations: int = 0
    clones: int = 0
    _hash: int | None = None
    team_events: dict[int, set[int]] = field(default_factory=dict)
    team_rounds: np.ndarray = field(default_factory=lambda: np.array([]))

    # Class variables
    ctx: ClassVar[ScheduleContext]

    def __post_init__(self) -> None:
        """Post-initialization to set up fitness array."""
        if self.schedule.size == 0:
            self.schedule = np.full(Schedule.ctx.n_total_events, -1, dtype=int)

        if not self.team_events:
            self.team_events = defaultdict(set)

        if self.team_rounds.size == 0:
            self.team_rounds = Schedule.ctx.teams_roundreqs_arr.copy()

    def __len__(self) -> int | np.signedinteger:
        """Return the number of scheduled events."""
        return np.count_nonzero(self.schedule >= 0)

    def get_size(self) -> int | np.signedinteger:
        """Return the number of scheduled events."""
        return self.__len__()

    def __eq__(self, other: object) -> bool:
        """Two Schedules are equal if they assign the same teams to the same events."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Hash is based on the frozenset of (event_id, team_id) pairs."""
        if self._hash is None:
            self._hash = hash(frozenset(frozenset(events) for events in self.team_events.values()))
        return self._hash

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        return Schedule(
            schedule=self.schedule.copy(),
            fitness=self.fitness.copy(),
            team_fitnesses=self.team_fitnesses.copy(),
            rank=self.rank,
            origin=self.origin,
            mutations=self.mutations,
            clones=self.clones + 1,
            _hash=self._hash,
            team_events={k: v.copy() for k, v in self.team_events.items()},
            team_rounds=self.team_rounds.copy(),
        )

    def swap_assignment(self, team: int, old_event: int, new_event: int) -> None:
        """Switch an event for a team in the schedule."""
        if team == -1:
            return

        self.unassign(team, old_event)
        self.assign(team, new_event)

    def assign(self, team: int, event: int) -> None:
        """Add an event to a team's scheduled events."""
        if team == -1:
            return

        roundtype = Schedule.ctx.event_props.roundtype_idx[event]
        self.team_events[team].add(event)
        self.team_rounds[team, roundtype] -= 1
        self.schedule[event] = team
        self._hash = None

    def unassign(self, team: int, event: int) -> None:
        """Remove an event from a team's scheduled events."""
        if team == -1:
            return

        roundtype = Schedule.ctx.event_props.roundtype_idx[event]
        self.team_events[team].remove(event)
        self.team_rounds[team, roundtype] += 1
        self.schedule[event] = -1
        self._hash = None

    def needs_round(self, team: int, roundtype: int) -> bool:
        """Check if a team still needs to participate in a given round type."""
        return self.team_rounds[team, roundtype] > 0

    def all_rounds_needed(self, roundtype: int) -> np.ndarray:
        """Return all teams that still need roundtype."""
        return (self.team_rounds[:, roundtype] > 0).nonzero()[0]

    def any_rounds_needed(self) -> bool:
        """Check if any team still needs rounds."""
        return self.team_rounds.sum() > 0

    def conflicts(self, team: int, new_event: int, *, ignore: int | None = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            team (int): The team to check for conflicts.
            new_event (int): The new event to check for conflicts.
            ignore (int): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        if team == -1:
            return False

        team_events = self.team_events[team]
        events_to_check = team_events

        if ignore is not None and ignore in team_events:
            events_to_check = events_to_check - {ignore}

        if new_event in events_to_check:
            return True

        return not events_to_check.isdisjoint(Schedule.ctx.conflict_map[new_event])

    def scheduled_events(self) -> np.ndarray:
        """Return the indices of scheduled events."""
        return (self.schedule >= 0).nonzero()[0]

    def unscheduled_events(self) -> np.ndarray:
        """Return the indices of unscheduled events."""
        return (self.schedule == -1).nonzero()[0]
