"""Represents a schedule (individual) with its associated fitness score."""

from collections import defaultdict
from collections.abc import Generator, ItemsView, Iterator, ValuesView
from dataclasses import dataclass, field

from ..data_model.event import Event
from ..data_model.team import Individual, Team, TeamMap

type Population = list[Schedule]


@dataclass(slots=True, order=True)
class Schedule:
    """Represents a schedule (individual) with its associated fitness score."""

    _teams: TeamMap = field(default_factory=dict, compare=False)
    _schedule: Individual = field(default_factory=dict, compare=False)
    fitness: tuple[float, ...] | None = field(default=None, compare=False)
    rank: int = field(default=9999, compare=True)
    crowding: float = field(default=0.0, compare=False)
    _cached_all_teams: list[Team] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _cached_matches: dict[str, list[tuple[Event, Event, Team, Team]]] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _hash: int = field(
        default=None,
        init=False,
        repr=False,
    )

    def __len__(self) -> int:
        """Return the number of scheduled events."""
        return len(self._schedule)

    def __getitem__(self, event: Event) -> Team:
        """Get the team assigned to a specific event."""
        try:
            return self._schedule[event]
        except KeyError:
            msg = f"The event {event} is not scheduled."
            raise KeyError(msg) from None

    def __setitem__(self, event: Event, team: Team) -> None:
        """Assign a team to a specific event."""
        self._schedule[event] = team
        self._cached_matches = None
        self._cached_all_teams = None
        self._hash = None

    def __contains__(self, event: Event) -> bool:
        """Check if a specific event is scheduled."""
        return event in self._schedule

    def __iter__(self) -> Generator[Event]:
        """Iterate over the Events in the schedule."""
        return iter(self._schedule)

    def __hash__(self) -> int:
        """Return a hash of the schedule based on its teams."""
        if self._hash is not None:
            return self._hash

        sched_hash = []
        sorted_schedule = sorted(self._schedule.items(), key=lambda x: x[0].identity)

        for e, t in sorted_schedule:
            sched_hash.extend([e.identity, t.identity])

        self._hash = hash(tuple(sched_hash))
        return self._hash

    def get_matches(self) -> dict[str, list[tuple[Event, Event, Team, Team]]]:
        """Get all matches in the schedule."""
        if self._cached_matches is not None:
            return self._cached_matches

        self._cached_matches = defaultdict(list)

        for event1, team1 in self._schedule.items():
            if not (event2 := event1.paired_event) or event1.location.side != 1:
                continue

            rt = event1.round_type

            if team2 := self._schedule[event2]:
                self._cached_matches[rt].append((event1, event2, team1, team2))

        return self._cached_matches

    def clone(self) -> "Schedule":
        """Create a deep copy of the Schedule instance."""
        new_teams = {identity: team.clone() for identity, team in self._teams.items()}
        new_individual = {event: new_teams[team.identity] for event, team in self._schedule.items()}
        return Schedule(new_teams, new_individual, self.fitness, self.rank, self.crowding)

    def all_teams(self) -> list[Team]:
        """Return a list of all teams in the schedule."""
        if self._cached_all_teams is None:
            self._cached_all_teams = list(self._teams.values())
        return self._cached_all_teams

    def keys(self) -> Iterator[Event]:
        """Return an iterator over the events (keys)."""
        yield from self._schedule.keys()

    def values(self) -> ValuesView[Team]:
        """Return an iterator over the assigned teams/matches (values)."""
        return self._schedule.values()

    def items(self) -> ItemsView[Event, Team]:
        """Return an iterator over the (event, team/match) pairs."""
        return self._schedule.items()

    def get_team(self, team: Team) -> Team:
        """Get a team object by its identity."""
        return self._teams[team.identity]
