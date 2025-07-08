"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from random import Random

from ..data_model.event import Event, EventFactory
from ..data_model.team import Team, TeamFactory
from ..genetic.schedule import Schedule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    team_factory: TeamFactory
    event_factory: EventFactory
    rng: Random


@dataclass(slots=True)
class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    events: list[Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.events = self.event_factory.flat_list()

    @abstractmethod
    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover."""

    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Crossover two parents to produce a child."""
        p1, p2 = parents
        child = Schedule(self.team_factory.build())
        p1_genes, p2_genes = self.get_genes(p1, p2)
        self._populate_from_parent(child, p1, p1_genes, first=True)
        self._populate_from_parent(child, p2, p2_genes)
        return child if child and self._repair_crossover(child) else None

    def _populate_from_parent(
        self, child: Schedule, parent: Schedule, events: Iterable[Event], *, first: bool = False
    ) -> None:
        """Populate the child individual from parent genes."""
        get_team_from_child = child.get_team
        for event1 in events:
            if (event2 := event1.paired_event) is None:
                team = get_team_from_child(parent[event1])
                if first or (not team.conflicts(event1) and team.needs_round(event1.round_type)):
                    self._populate_single(child, event1, team)
                continue

            if event2 is not None and event1.location.side != 1:
                continue

            team1 = get_team_from_child(parent[event1])
            team2 = get_team_from_child(parent[event2])

            if first or (
                not team1.conflicts(event1)
                and not team2.conflicts(event2)
                and team1.needs_round(event1.round_type)
                and team2.needs_round(event2.round_type)
            ):
                self._populate_match(child, event1, event2, team1, team2)

    def _populate_single(self, child: Schedule, event: Event, team: Team) -> None:
        """Populate a single event in the child schedule."""
        team.add_event(event)
        child[event] = team

    def _populate_match(self, child: Schedule, event1: Event, event2: Event, team1: Team, team2: Team) -> None:
        """Populate a match event in the child schedule."""
        team1.add_event(event1)
        team2.add_event(event2)
        team1.add_opponent(team2)
        team2.add_opponent(team1)
        child[event1] = team1
        child[event2] = team2

    def _repair_crossover(self, child: Schedule) -> bool:
        """Repair conflicts in the child individual by finding new slots for conflicted teams."""
        rt_teams_needed = {}
        for rc in self.event_factory.config.rounds:
            rt_teams_needed[rc.round_type] = rc.teams_per_round

        open_events = set(self.events) - child.keys()
        open_events_by_rt = defaultdict(list)
        for e in open_events:
            if (e.paired_event is not None and e.location.side == 1) or e.paired_event is None:
                open_events_by_rt[(e.round_type, rt_teams_needed[e.round_type])].append(e)

        needs_by_rt = defaultdict(list)
        for team in child.all_teams():
            for rt, n in team.round_types.items():
                if n > 0:
                    for _ in range(n):
                        needs_by_rt[(rt, rt_teams_needed[rt])].append(team)

        for (rt, n), teams in needs_by_rt.items():
            if n == 1:
                self._assign_singles(teams, open_events_by_rt[(rt, n)], child)
            elif n == 2:
                self._assign_matches(teams, open_events_by_rt[(rt, n)], child)

        return all(t.rounds_needed() == 0 for t in child.all_teams())

    def _assign_singles(self, teams: list[Team], open_events: list[Event], child: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        self.rng.shuffle(teams)
        self.rng.shuffle(open_events)
        for team in teams:
            for i, event in enumerate(open_events):
                if not team.conflicts(event):
                    self._populate_single(child, event, team)
                    open_events.pop(i)
                    break

    def _assign_matches(self, teams: list[Team], open_events: list[Event], child: Schedule) -> None:
        """Assign match events to teams that need them."""
        self.rng.shuffle(teams)
        self.rng.shuffle(open_events)
        for team1, team2 in zip(teams[::2], teams[1::2], strict=True):
            if team1.identity == team2.identity:
                return

            for j, e1 in enumerate(open_events):
                e2 = e1.paired_event

                if not team1.conflicts(e1) and not team2.conflicts(e2):
                    self._populate_match(child, e1, e2, team1, team2)
                    open_events.pop(j)
                    break

                if not team1.conflicts(e2) and not team2.conflicts(e1):
                    self._populate_match(child, e2, e1, team2, team1)
                    open_events.pop(j)
                    break


@dataclass(slots=True)
class KPoint(EventCrossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = field(default=1)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(KPoint, self).__post_init__()

        if not 1 <= self.k < len(self.events):
            msg = "k must be between 1 and the number of events."
            raise ValueError(msg)

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover."""
        evts = self.events
        ne = len(evts)
        indices = sorted(self.rng.sample(range(1, ne), self.k))
        genes = []
        start = 0
        for i in indices:
            genes.append(evts[start:i])
            start = i
        genes.append(evts[start:])
        ng = len(genes)
        p1_genes = (genes[i][x] for i in range(ng) if i % 2 == 0 for x in range(len(genes[i])) if genes[i][x] in p1)
        p2_genes = (genes[i][x] for i in range(ng) if i % 2 == 1 for x in range(len(genes[i])) if genes[i][x] in p2)
        return p1_genes, p2_genes


@dataclass(slots=True)
class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover."""
        evts = self.events
        ne = len(evts)
        indices = self.rng.sample(range(ne), ne)
        mid = ne // 2
        p1_genes = (evts[i] for i in indices[:mid] if evts[i] in p1)
        p2_genes = (evts[i] for i in indices[mid:] if evts[i] in p2)
        return p1_genes, p2_genes


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover."""
        evts = self.events
        ne = len(evts)
        indices = [1 if self.rng.uniform(0, 1) < 0.5 else 2 for _ in range(ne)]
        p1_genes = (evts[i] for i in range(ne) if indices[i] == 1 and evts[i] in p1)
        p2_genes = (evts[i] for i in range(ne) if indices[i] == 2 and evts[i] in p2)
        return p1_genes, p2_genes
