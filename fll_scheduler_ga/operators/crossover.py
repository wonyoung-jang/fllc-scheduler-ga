"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from random import Random

from ..data_model.event import Event, EventFactory
from ..data_model.team import TeamFactory
from ..genetic.schedule import Schedule
from ..genetic.schedule_repairer import ScheduleRepairer

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

    repairer: ScheduleRepairer
    events: list[Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.events = self.event_factory.flat_list()

    @abstractmethod
    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover."""

    def crossover(self, parents: list[Schedule]) -> Schedule | None:
        """Crossover two parents to produce a child."""
        p1, p2 = self.rng.sample(parents, k=2)

        if child1 := self._produce_child(p1, p2):
            return child1

        if child2 := self._produce_child(p2, p1):
            return child2

        return None

    def _produce_child(self, p1: Schedule, p2: Schedule) -> Schedule | None:
        """Produce a child schedule from two parents."""
        child = Schedule(self.team_factory.build())
        p1_genes, p2_genes = self.get_genes(p1, p2)
        self._transfer_genes(child, p1, p1_genes, first=True)
        self._transfer_genes(child, p2, p2_genes)
        return child if child and self.repairer.repair(child) else None

    def _transfer_genes(
        self, child: Schedule, parent: Schedule, events: Iterable[Event], *, first: bool = False
    ) -> None:
        """Populate the child individual from parent genes."""
        get_team_from_child = child.get_team
        for event1 in events:
            if (event2 := event1.paired_event) is None:
                team = get_team_from_child(parent[event1])
                if first or (not team.conflicts(event1) and team.needs_round(event1.round_type)):
                    child[event1] = team
                continue

            if event1.location.side != 1:
                continue

            team1 = get_team_from_child(parent[event1])
            team2 = get_team_from_child(parent[event2])

            if first or not (
                team1.conflicts(event1)
                or team2.conflicts(event2)
                or not team1.needs_round(event1.round_type)
                or not team2.needs_round(event2.round_type)
            ):
                child.add_match(event1, event2, team1, team2)


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
