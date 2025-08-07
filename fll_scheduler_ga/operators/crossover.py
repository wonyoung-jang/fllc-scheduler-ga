"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from random import Random

from ..config.app_config import AppConfig
from ..config.constants import CrossoverOp
from ..data_model.event import Event, EventFactory
from ..data_model.schedule import Schedule
from ..data_model.team import TeamFactory

logger = logging.getLogger(__name__)


def build_crossovers(
    app_config: AppConfig, team_factory: TeamFactory, event_factory: EventFactory
) -> Iterator["Crossover"]:
    """Build and return a tuple of crossover operators based on the configuration."""
    rng = app_config.rng
    variant_map = {
        CrossoverOp.K_POINT: lambda k: KPoint(team_factory, event_factory, rng, k=k),
        CrossoverOp.SCATTERED: lambda: Scattered(team_factory, event_factory, rng),
        CrossoverOp.UNIFORM: lambda: Uniform(team_factory, event_factory, rng),
        CrossoverOp.ROUND_TYPE_CROSSOVER: lambda: RoundTypeCrossover(team_factory, event_factory, rng),
        CrossoverOp.PARTIAL_CROSSOVER: lambda: PartialCrossover(team_factory, event_factory, rng),
    }

    if not (crossover_types := app_config.operators.crossover_types):
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return

    for variant_name in crossover_types:
        if variant_name not in variant_map:
            msg = f"Unknown crossover type in config: {variant_name}"
            raise ValueError(msg)
        else:
            crossover_factory = variant_map[variant_name]
            if variant_name == CrossoverOp.K_POINT:
                for k in app_config.operators.crossover_ks:
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
        self.events = self.event_factory.as_list()

    @abstractmethod
    def crossover(self, parents: tuple[Schedule]) -> Iterator[Schedule]:
        """Crossover two parents to produce a child.

        Args:
            parents (tuple[Schedule]): List of parent schedules.

        Yields:
            Schedule: The child schedule produced from the crossover.

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

    def crossover(self, parents: tuple[Schedule]) -> Iterator[Schedule]:
        """Crossover two parents to produce a child."""
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
        p1_genes, p2_genes = self.get_genes(p1, p2)
        self._transfer_genes_from_parent1(child, p1, p1_genes)
        self._transfer_genes_from_parent2(child, p2, p2_genes)
        return child

    def _transfer_genes_from_parent1(self, child: Schedule, parent: Schedule, events: Iterator[Event]) -> None:
        """Transfer genes from the first parent. Fewer checks needed."""
        get_team_from_child = child.get_team
        for e1 in events:
            t1 = get_team_from_child(parent[e1].identity)
            if (e2 := e1.paired) and e1.location.side == 1:
                t2 = get_team_from_child(parent[e2].identity)
                child.assign_match(e1, e2, t1, t2)
            elif e2 is None:
                child.assign_single(e1, t1)

    def _transfer_genes_from_parent2(self, child: Schedule, parent: Schedule, events: Iterator[Event]) -> None:
        """Transfer genes from the second parent."""
        get_team_from_child = child.get_team
        for e1 in events:
            t1 = get_team_from_child(parent[e1].identity)
            if (e2 := e1.paired) and e1.location.side == 1:
                t2 = get_team_from_child(parent[e2].identity)
                if (
                    not t1.needs_round(e1.roundtype)
                    or not t2.needs_round(e2.roundtype)
                    or t1.conflicts(e1)
                    or t2.conflicts(e2)
                ):
                    continue
                child.assign_match(e1, e2, t1, t2)
            elif e2 is None:
                if not t1.needs_round(e1.roundtype) or t1.conflicts(e1):
                    continue
                child.assign_single(e1, t1)


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
        yield (e for e in evts for i, r in enumerate(rt) if e.roundtype == r and i % 2 != 0 and e in p1)
        yield (e for e in evts for i, r in enumerate(rt) if e.roundtype == r and i % 2 == 0 and e in p2)


@dataclass(slots=True)
class PartialCrossover(EventCrossover):
    """Partial crossover operator for genetic algorithms.

    This operator takes a random subset of events from each parent.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for Partial crossover."""
        evts = self.events
        ne = len(evts)
        sections = sorted(self.rng.sample(range(1, ne), 4))
        indices = self.rng.sample(range(ne), ne)
        yield (evts[i] for i in indices[: sections[0]] if evts[i] in p1)
        yield (evts[i] for i in indices[sections[-1] :] if evts[i] in p2)
