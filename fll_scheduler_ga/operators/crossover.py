"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from random import Random

from ..config.config import RoundType
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
    events: list[Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.events = [e for el in self.event_factory.build().values() for e in el]

    @abstractmethod
    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Crossover two parents to produce a child."""

    def _populate_from_parent(
        self, child: Schedule, parent: Schedule, events: Iterable[Event], conflicted: list[Team]
    ) -> None:
        """Populate the child individual from parent genes."""
        for event in (e for e in events if e in parent):
            if event.paired_event is None:
                team = child.get_team(parent[event])
                if team.conflicts(event) or not team.needs_round(event):
                    conflicted.append(team)
                else:
                    team.add_event(event)
                    child[event] = team
                continue

            if event.paired_event is not None and event.location.side != 1:
                continue

            team1 = child.get_team(parent[event])
            team2 = child.get_team(parent[event.paired_event])
            if (
                team1.conflicts(event)
                or team2.conflicts(event.paired_event)
                or not team1.needs_round(event)
                or not team2.needs_round(event.paired_event)
            ):
                conflicted.extend([team1, team2])
            else:
                team1.add_event(event)
                team2.add_event(event.paired_event)
                team1.add_opponent(team2)
                team2.add_opponent(team1)
                child[event] = team1
                child[event.paired_event] = team2

    def _repair_crossover(self, child: Schedule, conflicted: list[Team]) -> bool:
        """Repair conflicts in the child individual by finding new slots for conflicted teams."""
        if not conflicted:
            return True

        open_events = set(self.events) - child.keys()
        open_single_events = defaultdict(list)
        open_match_events = defaultdict(list)
        for e in open_events:
            if e.paired_event is not None:
                if e.location.side == 1:
                    open_match_events[e.round_type].append(e)
            else:
                open_single_events[e.round_type].append(e)

        conflicted_teams = list(set(conflicted))
        self.rng.shuffle(conflicted_teams)

        repair_results = []
        for team in conflicted_teams:
            result = self._find_and_book_slot(child, team, open_single_events, open_match_events, conflicted_teams)
            repair_results.append(result)

        return all(repair_results)

    def _find_and_book_slot(
        self,
        child: Schedule,
        team1: Team,
        open_singles: dict[RoundType, list[Event]],
        open_matches: dict[RoundType, list[Event]],
        conflicted_teams: list[Team],
    ) -> bool:
        """Find and book a new slot for a team in the child individual."""
        while team1.rounds_needed > 0:
            rt = next((rt for rt in team1.round_types if team1.needs_round(rt)), None)
            if rt is None:
                return True

            if self._add_single(
                child,
                team1,
                open_singles.get(rt, []),
            ) or self._add_match(
                child,
                team1,
                open_matches.get(rt, []),
                conflicted_teams,
            ):
                continue
            return False
        return True

    def _add_single(self, child: Schedule, team: Team, open_slots: list[Event]) -> bool:
        """Add a single-team event to the child schedule."""
        for i, event in enumerate(open_slots):
            if not team.conflicts(event):
                team.add_event(event)
                child[event] = team
                open_slots.pop(i)
                return True
        return False

    def _add_match(self, child: Schedule, team1: Team, open_slots: list[Event], conflicted_teams: list[Team]) -> bool:
        rt = next(rt for rt in team1.round_types if team1.needs_round(rt))
        partner_pool = [t for t in conflicted_teams if t != team1 and t.needs_round(rt)]
        if not partner_pool:
            partner_pool = [t for t in child.all_teams if t != team1 and t.needs_round(rt)]
        if not partner_pool:
            return False
        team2 = self.rng.choice(partner_pool)

        for i, e1 in enumerate(open_slots):
            e2 = e1.paired_event
            if not team1.conflicts(e1) and not team2.conflicts(e2):
                team1.add_event(e1)
                team2.add_event(e2)
                team1.add_opponent(team2)
                team2.add_opponent(team1)
                child[e1] = team1
                child[e2] = team2
                open_slots.pop(i)
                return True
            if not team1.conflicts(e2) and not team2.conflicts(e1):
                team1.add_event(e2)
                team2.add_event(e1)
                team1.add_opponent(team2)
                team2.add_opponent(team1)
                child[e2] = team1
                child[e1] = team2
                open_slots.pop(i)
                return True
        return False


@dataclass(slots=True)
class KPoint(Crossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = 1

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(KPoint, self).__post_init__()

        if not 1 <= self.k < len(self.events):
            msg = "k must be between 1 and the number of events."
            raise ValueError(msg)

    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Perform k-point crossover."""
        p1, p2 = parents
        child = Schedule(self.team_factory.build())
        conflicted = []
        indices = sorted(self.rng.sample(range(1, len(self.events)), self.k))
        genes = []
        start = 0
        for i in indices:
            genes.append(self.events[start:i])
            start = i
        genes.append(self.events[start:])
        for i, gene in enumerate(genes):
            parent = p1 if i % 2 == 0 else p2
            self._populate_from_parent(child, parent, gene, conflicted)
        return child if child and self._repair_crossover(child, conflicted) else None


@dataclass(slots=True)
class Scattered(Crossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Perform scattered crossover."""
        p1, p2 = parents
        child = Schedule(self.team_factory.build())
        conflicted = []
        indices = self.rng.sample(range(len(self.events)), len(self.events))
        mid = len(self.events) // 2 + 1 if len(self.events) % 2 else len(self.events) // 2
        p1_genes = (self.events[i] for i in indices[:mid])
        p2_genes = (self.events[i] for i in indices[mid:])
        self._populate_from_parent(child, p1, p1_genes, conflicted)
        self._populate_from_parent(child, p2, p2_genes, conflicted)
        return child if child and self._repair_crossover(child, conflicted) else None


@dataclass(slots=True)
class Uniform(Crossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    """

    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Perform uniform crossover."""
        p1, p2 = parents
        child = Schedule(self.team_factory.build())
        conflicted = []
        indices = [1 if self.rng.uniform(0, 1) < 0.5 else 2 for _ in range(len(self.events))]
        p1_genes = (self.events[i] for i in range(len(self.events)) if indices[i] == 1)
        p2_genes = (self.events[i] for i in range(len(self.events)) if indices[i] == 2)
        self._populate_from_parent(child, p1, p1_genes, conflicted)
        self._populate_from_parent(child, p2, p2_genes, conflicted)
        return child if child and self._repair_crossover(child, conflicted) else None
