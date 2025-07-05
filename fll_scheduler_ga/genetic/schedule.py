"""Represents a schedule (individual) with its associated fitness score."""

from collections.abc import Generator, ItemsView, KeysView, ValuesView
from copy import deepcopy
from dataclasses import dataclass, field

from ..data_model.event import Event
from ..data_model.team import Individual, Team, TeamMap

type Population = list[Schedule]


@dataclass(slots=True, order=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    _teams: TeamMap = field(default_factory=dict, compare=False)
    _schedule: Individual = field(default_factory=dict, compare=False)
    fitness: tuple[float, ...] = field(default=None, compare=False)
    rank: int = field(default=9999, compare=True)
    crowding: float = field(default=0.0, compare=False)

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return len(self._schedule)

    def __getitem__(self, event: Event) -> Team:
        """Get the team or teams assigned to a specific event."""
        try:
            return self._schedule[event]
        except KeyError:
            msg = f"The event {event} is not scheduled."
            raise KeyError(msg) from None

    def __setitem__(self, event: Event, teams: Team) -> None:
        """Assign a team or teams to a specific event."""
        self._schedule[event] = teams

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return event in self._schedule

    def __iter__(self) -> Generator[Event]:
        """Iterate over the Events in the schedule."""
        return iter(self._schedule)

    def __deepcopy__(self, memo: dict[int, object]) -> "Schedule":
        """Create a deep copy of the Schedule instance."""
        if id(self) in memo:
            return memo[id(self)]

        new_teams = {identity: deepcopy(team) for identity, team in self._teams.items()}
        new_individual = {}
        for event, booked_item in self.items():
            new_individual[event] = new_teams[booked_item.identity]

        new_schedule = Schedule(
            new_teams,
            new_individual,
            fitness=self.fitness,
            rank=self.rank,
            crowding=self.crowding,
        )
        memo[id(self)] = new_schedule

        return new_schedule

    @property
    def all_teams(self) -> list[Team]:
        """Return a list of all teams in the schedule."""
        return list(self._teams.values())

    def keys(self) -> KeysView[Event]:
        """Return an iterator over the events (keys)."""
        return self._schedule.keys()

    def values(self) -> ValuesView[Team]:
        """Return an iterator over the assigned teams/matches (values)."""
        return self._schedule.values()

    def items(self) -> ItemsView[Event, Team]:
        """Return an iterator over the (event, team/match) pairs."""
        return self._schedule.items()

    def get_team(self, team: Team) -> Team:
        """Get a team object by its identity."""
        return self._teams[team.identity]
