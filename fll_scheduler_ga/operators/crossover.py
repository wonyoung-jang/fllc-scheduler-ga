"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from random import Random

from ..config.constants import CrossoverOps
from ..config.ga_operators_config import OperatorConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import TeamFactory
from ..genetic.schedule import Schedule

logger = logging.getLogger(__name__)


def build_crossovers(
    o_config: OperatorConfig,
    rng: Random,
    team_factory: TeamFactory,
    event_factory: EventFactory,
) -> Iterator["Crossover"]:
    """Build and return a tuple of crossover operators based on the configuration."""
    variant_map = {
        CrossoverOps.K_POINT: lambda k: KPoint(team_factory, event_factory, rng, k=k),
        CrossoverOps.SCATTERED: lambda: Scattered(team_factory, event_factory, rng),
        CrossoverOps.UNIFORM: lambda: Uniform(team_factory, event_factory, rng),
        CrossoverOps.ROUND_TYPE_CROSSOVER: lambda: RoundTypeCrossover(team_factory, event_factory, rng),
        CrossoverOps.PARTIAL_CROSSOVER: lambda: PartialCrossover(team_factory, event_factory, rng),
    }

    if not o_config.crossover_types:
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return

    for variant_name in o_config.crossover_types:
        if variant_name not in variant_map:
            msg = f"Unknown crossover type in config: {variant_name}"
            raise ValueError(msg)
        else:
            crossover_factory = variant_map[variant_name]
            if variant_name == CrossoverOps.K_POINT:
                for k in o_config.crossover_ks:
                    if k <= 0:
                        msg = f"Invalid crossover k value: {k}. Must be greater than 0."
                        raise ValueError(msg)
                    yield crossover_factory(k)
            else:
                yield crossover_factory()


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    team_factory: TeamFactory
    event_factory: EventFactory
    rng: Random
    events: list[Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the crossover operator."""
        self.events = self.event_factory.flat_list()

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
    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Yields:
            Iterator[Event]: Genes from each parents.

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
        yield self._produce_child(p1, p2)
        yield self._produce_child(p2, p1)

    def _produce_child(self, p1: Schedule, p2: Schedule) -> Schedule:
        """Produce a child schedule from two parents.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Returns:
            Schedule : The child schedule produced from crossover.

        """
        child = Schedule(self.team_factory.build())
        parent_genes = self.get_genes(p1, p2)
        self._transfer_genes(child, p1, next(parent_genes), first=True)
        self._transfer_genes(child, p2, next(parent_genes), first=False)
        return child

    def _transfer_genes(self, child: Schedule, parent: Schedule, events: Iterator[Event], *, first: bool) -> None:
        """Populate the child individual from parent genes.

        Args:
            child (Schedule): The child schedule to populate.
            parent (Schedule): The parent schedule to copy genes from.
            events (Iterable[Event]): The events to copy.
            first (bool, optional): Whether this is the first parent.

        """
        get_team_from_child = child.get_team

        for e1 in events:
            if (e2 := e1.paired_event) and e1.location.side == 1:
                t1 = get_team_from_child(parent[e1].identity)
                t2 = get_team_from_child(parent[e2].identity)
                if first or (
                    t1.needs_round(e1.round_type)
                    and t2.needs_round(e2.round_type)
                    and not t1.conflicts(e1)
                    and not t2.conflicts(e2)
                ):
                    child.assign_match(e1, e2, t1, t2)
            elif e2 is None:
                team = get_team_from_child(parent[e1].identity)
                if first or (team.needs_round(e1.round_type) and not team.conflicts(e1)):
                    child.assign_single(e1, team)


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

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for KPoint crossover."""
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
        yield (genes[i][x] for i in range(ng) if i % 2 == 0 for x in range(len(genes[i])) if genes[i][x] in p1)
        yield (genes[i][x] for i in range(ng) if i % 2 == 1 for x in range(len(genes[i])) if genes[i][x] in p2)


@dataclass(slots=True)
class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for Scattered crossover."""
        evts = self.events
        ne = len(evts)
        mid = ne // 2
        indices = self.rng.sample(range(ne), ne)
        yield (evts[i] for i in indices[:mid] if evts[i] in p1)
        yield (evts[i] for i in indices[mid:] if evts[i] in p2)


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for Uniform crossover."""
        evts = self.events
        ne = len(evts)
        indices = [1 if self.rng.choice([True, False]) else 2 for _ in range(ne)]
        yield (evts[i] for i in range(ne) if indices[i] == 1 and evts[i] in p1)
        yield (evts[i] for i in range(ne) if indices[i] == 2 and evts[i] in p2)


@dataclass(slots=True)
class RoundTypeCrossover(EventCrossover):
    """Round type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for RoundType crossover."""
        evts = self.events
        rt = list(self.event_factory.config.round_requirements.keys())
        yield (e for e in evts for i, r in enumerate(rt) if e.round_type == r and i % 2 != 0 and e in p1)
        yield (e for e in evts for i, r in enumerate(rt) if e.round_type == r and i % 2 == 0 and e in p2)


@dataclass(slots=True)
class PartialCrossover(EventCrossover):
    """Partial crossover operator for genetic algorithms.

    This operator takes a random subset of events from each parent.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for Partial crossover."""
        evts = self.events
        ne = len(evts)
        thirds = sorted(self.rng.sample(range(1, ne), 3))
        indices = self.rng.sample(range(ne), ne)
        yield (evts[i] for i in indices[: thirds[0]] if evts[i] in p1)
        yield (evts[i] for i in indices[thirds[2] :] if evts[i] in p2)
