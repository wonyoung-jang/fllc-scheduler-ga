"""Represents a schedule (individual) with its associated fitness score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .event import Event
from .team import Team

type Population = list[Schedule]
type Individual = dict[Event, int]
type Match = tuple[Event, Event, Team, Team]


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

    team_identities: ClassVar[dict[int, int | str]]
    total_num_events: ClassVar[int]

    def __post_init__(self) -> None:
        """Post-initialization to set up fitness array."""
        if self.schedule is None:
            self.schedule = np.full(Schedule.total_num_events, -1, dtype=int)

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

    def __delitem__(self, event: Event) -> None:
        """Remove a specific event from the schedule."""
        self.schedule[event.idx] = -1

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
        return hash(self.schedule.tobytes())

    @classmethod
    def set_team_identities(cls, identities: dict[int, int | str]) -> None:
        """Set the team identities for the schedule."""
        cls.team_identities = identities

    @classmethod
    def set_total_num_events(cls, total: int) -> None:
        """Set the total number of events for the schedule."""
        cls.total_num_events = total

    def to_array(self) -> np.ndarray:
        """Convert the schedule to its core numpy array representation."""
        return self.schedule

    def clone(self) -> Schedule:
        """Create a deep copy of the schedule."""
        return Schedule(
            teams=np.array([t.clone() for t in self.teams]),
            schedule=self.schedule.copy() if self.schedule is not None else None,
            fitness=self.fitness.copy() if self.fitness is not None else None,
            team_fitnesses=self.team_fitnesses.copy() if self.team_fitnesses is not None else None,
            origin=self.origin,
            mutations=self.mutations,
            clones=self.clones + 1,
        )

    def scheduled_event_indices(self) -> np.ndarray:
        """Return the indices of scheduled events."""
        return np.where(self.schedule > -1)[0]

    def unscheduled_event_indices(self) -> np.ndarray:
        """Return the indices of unscheduled events."""
        return np.where(self.schedule == -1)[0]

    def assign_single(self, event: Event, team: Team) -> None:
        """Assign a single-team event to a team."""
        team.add_event(event)
        self[event] = team

    def assign_match(self, event1: Event, event2: Event, team1: Team, team2: Team) -> None:
        """Assign a match event to two teams."""
        team1.add_event(event1)
        team2.add_event(event2)
        self[event1] = team1
        self[event2] = team2

    def destroy_event(self, event: Event) -> None:
        """Destroy an event, whether single or match."""
        if event.paired:
            self.destroy_match(event, event.paired)
        else:
            self.destroy_single(event)

    def destroy_single(self, event: Event) -> None:
        """Destroy a single-team event."""
        if team := self[event]:
            team.remove_event(event)
            del self[event]

    def destroy_match(self, event1: Event, event2: Event) -> None:
        """Destroy a match event."""
        if (team1 := self[event1]) and (team2 := self[event2]):
            team1.remove_event(event1)
            team2.remove_event(event2)
            del self[event1]
            del self[event2]

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
