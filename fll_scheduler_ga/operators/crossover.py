"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from random import Random

from ..data_model.event import Event
from ..data_model.team import TeamFactory
from ..genetic.schedule import Schedule
from .repairer import Repairer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    team_factory: TeamFactory
    events: list[Event]
    rng: Random
    repairer: Repairer

    @abstractmethod
    def crossover(self, parents: list[Schedule]) -> list[Schedule]:
        """Crossover two parents to produce a child.

        Args:
            parents (list[Schedule]): List of parent schedules.

        Returns:
            list[Schedule]: The child schedule produced from the crossover, or None if unsuccessful.

        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)


@dataclass(slots=True)
class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    @abstractmethod
    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], Iterable[Event]]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], Iterable[Event]]: Genes from both parents.

        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def crossover(self, parents: list[Schedule]) -> Iterator[Schedule]:
        """Crossover two parents to produce a child.

        Args:
            parents (list[Schedule]): List of parent schedules.

        Yields:
            Schedule: The child schedule produced from the crossover.

        """
        p1, p2 = parents

        if child1 := self._produce_child(p1, p2):
            yield child1

        if child2 := self._produce_child(p2, p1):
            yield child2

    def _produce_child(self, p1: Schedule, p2: Schedule) -> Schedule | None:
        """Produce a child schedule from two parents.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            Schedule | None: The child schedule produced from the crossover, or None if unsuccessful.

        """
        child = Schedule(self.team_factory.build())
        p1_genes, p2_genes = self.get_genes(p1, p2)
        self._transfer_genes(child, p1, p1_genes, first=True)
        self._transfer_genes(child, p2, p2_genes)
        return self.repairer.repair(child)

    def _transfer_genes(
        self, child: Schedule, parent: Schedule, events: Iterable[Event], *, first: bool = False
    ) -> None:
        """Populate the child individual from parent genes.

        Args:
            child (Schedule): The child schedule to populate.
            parent (Schedule): The parent schedule to copy genes from.
            events (Iterable[Event]): The events to copy.
            first (bool, optional): Whether this is the first parent. Defaults to False.

        """
        for event1 in events:
            if (event2 := event1.paired_event) and event1.location.side == 1:
                team1 = child.get_team(parent[event1].identity)
                team2 = child.get_team(parent[event2].identity)
                if first or (
                    team1.needs_round(event1.round_type)
                    and team2.needs_round(event2.round_type)
                    and not team1.conflicts(event1)
                    and not team2.conflicts(event2)
                ):
                    team1.add_event(event1)
                    team2.add_event(event2)
                    team1.add_opponent(team2)
                    team2.add_opponent(team1)
                    child[event1] = team1
                    child[event2] = team2
            elif event2 is None:
                team = child.get_team(parent[event1].identity)
                if first or (team.needs_round(event1.round_type) and not team.conflicts(event1)):
                    team.add_event(event1)
                    child[event1] = team


@dataclass(slots=True)
class KPoint(EventCrossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = field(default=1)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        if not 1 <= self.k < len(self.events):
            msg = "k must be between 1 and the number of events."
            raise ValueError(msg)

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], ...]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], ...]: Genes from both parents.

        """
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

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], ...]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], ...]: Genes from both parents.

        """
        evts = self.events
        ne = len(evts)
        mid = ne // 2
        indices = self.rng.sample(range(ne), ne)
        p1_genes = (evts[i] for i in indices[:mid] if evts[i] in p1)
        p2_genes = (evts[i] for i in indices[mid:] if evts[i] in p2)
        return p1_genes, p2_genes


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], ...]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], ...]: Genes from both parents.

        """
        evts = self.events
        ne = len(evts)
        indices = (1 if self.rng.uniform(0, 1) < 0.5 else 2 for _ in range(ne))
        p1_genes = (evts[i] for i in range(ne) if next(indices, 0) == 1 and evts[i] in p1)
        p2_genes = (evts[i] for i in range(ne) if next(indices, 0) == 2 and evts[i] in p2)
        return p1_genes, p2_genes


@dataclass(slots=True)
class RoundTypeCrossover(EventCrossover):
    """Round type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], ...]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], ...]: Genes from both parents.

        """
        evts = self.events
        teams = p1.all_teams()
        rt = list(teams[0].round_types.keys())
        p1_genes = (e for e in evts for i, r in enumerate(rt) if e.round_type == r and i % 2 != 0 and e in p1)
        p2_genes = (e for e in evts for i, r in enumerate(rt) if e.round_type == r and i % 2 == 0 and e in p2)
        return p1_genes, p2_genes


@dataclass(slots=True)
class PartialCrossover(EventCrossover):
    """Partial crossover operator for genetic algorithms.

    This operator takes a random subset of events from each parent.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> tuple[Iterable[Event], ...]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            tuple[Iterable[Event], ...]: Genes from both parents.

        """
        evts = self.events
        ne = len(evts)
        thirds = sorted(self.rng.sample(range(1, ne), 3))
        indices = self.rng.sample(range(ne), ne)
        p1_genes = (evts[i] for i in indices[: thirds[0]] if evts[i] in p1)
        p2_genes = (evts[i] for i in indices[thirds[2] :] if evts[i] in p2)
        return p1_genes, p2_genes
