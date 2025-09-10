"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import CrossoverOp
from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Iterator
    from random import Random

    from ..config.app_config import AppConfig
    from ..data_model.event import Event, EventFactory
    from ..data_model.team import Team, TeamFactory

logger = getLogger(__name__)


def build_crossovers(
    app_config: AppConfig, team_factory: TeamFactory, event_factory: EventFactory
) -> Iterator[Crossover]:
    """Build and return a tuple of crossover operators based on the configuration."""
    rng = app_config.rng
    crossovers = {
        CrossoverOp.K_POINT: lambda k: KPoint(team_factory, event_factory, rng, k=k),
        CrossoverOp.SCATTERED: lambda: Scattered(team_factory, event_factory, rng),
        CrossoverOp.UNIFORM: lambda: Uniform(team_factory, event_factory, rng),
        CrossoverOp.PARTIAL: lambda: Partial(team_factory, event_factory, rng),
        CrossoverOp.ROUND_TYPE_CROSSOVER: lambda: RoundTypeCrossover(team_factory, event_factory, rng),
        CrossoverOp.TIMESLOT_CROSSOVER: lambda: TimeSlotCrossover(team_factory, event_factory, rng),
        CrossoverOp.LOCATION_CROSSOVER: lambda: LocationCrossover(team_factory, event_factory, rng),
        CrossoverOp.BEST_TEAM_CROSSOVER: lambda: BestTeamCrossover(team_factory, event_factory, rng),
    }

    if not (crossover_types := app_config.operators.crossover_types):
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return

    for crossover_name in crossover_types:
        if crossover_name not in crossovers:
            msg = f"Unknown crossover type in config: {crossover_name}"
            raise ValueError(msg)
        elif crossover_name == CrossoverOp.K_POINT:
            if crossover_ks := app_config.operators.crossover_ks:
                for k in crossover_ks:
                    if k <= 0:
                        msg = f"Invalid crossover k value: {k}. Must be greater than 0."
                        raise ValueError(msg)
                    yield crossovers[crossover_name](k)
        else:
            yield crossovers[crossover_name]()


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    team_factory: TeamFactory
    event_factory: EventFactory
    rng: Random

    @abstractmethod
    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents.

        Args:
            parents (Iterator[Schedule]): An iterator containing the first and second parent schedules.

        Yields:
            Schedule : The child schedule produced from crossover.

        """

    def _create_child(
        self,
        p1: Schedule,
        p2: Schedule,
        p1_genes: Iterator[Event],
        p2_genes: Iterator[Event],
    ) -> Schedule:
        """Create a child schedule from two parents."""
        c = Schedule(teams=self.team_factory.build(), origin="Crossover")
        self._transfer_genes(c, p1, p2, p1_genes, p2_genes)
        c.clear_cache()
        return c

    def _transfer_genes(
        self, c: Schedule, p1: Schedule, p2: Schedule, p1_genes: Iterator[Event], p2_genes: Iterator[Event]
    ) -> None:
        """Transfer genes from both parents."""
        evts_from_p1 = ((e, e.paired, c.get_team(p1[e]), c.get_team(p1[e.paired])) for e in p1_genes)
        evts_from_p2 = ((e, e.paired, c.get_team(p2[e]), c.get_team(p2[e.paired])) for e in p2_genes)
        while True:
            from_p1 = next(evts_from_p1, None)
            from_p2 = next(evts_from_p2, None)
            if from_p1 is None and from_p2 is None:
                break

            self._assign_genes(c, from_p1)
            self._assign_genes(c, from_p2)

    def _assign_genes(self, c: Schedule, data: tuple[Event, Event, Team, Team] | None) -> None:
        """Assign genes."""
        if data is None:
            return

        e1, e2, t1, t2 = data

        if t1 is None or not t1.needs_round(e1.roundtype) or t1.conflicts(e1):
            return

        if e2 is None:
            c.assign_single(e1, t1)
            return

        if e1.location.side != 1:
            return

        if t2 is None or not t2.needs_round(e2.roundtype) or t2.conflicts(e2):
            return

        c.assign_match(e1, e2, t1, t2)


@dataclass(slots=True)
class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    events: list[Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the crossover operator."""
        self.events = self.event_factory.build()

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        if hasattr(self, "k"):
            return f"{self.__class__.__name__}(k={self.k})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for the crossover.

        Yields:
            Iterator[Event]: Genes for each parents.

        """

    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents."""
        p1, p2 = parents
        p1_genes, p2_genes = self.get_genes()
        yield self._create_child(p1, p2, p1_genes, p2_genes)
        yield self._create_child(p2, p1, p2_genes, p1_genes)


@dataclass(slots=True)
class TeamCrossover(Crossover):
    """Abstract base class for team based crossover operators in the FLL Scheduler GA."""

    event_map: dict[int, Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the event map."""
        self.event_map = self.event_factory.as_mapping()

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for the crossover.

        Args:
            p1 (Schedule): First parent schedule.
            p2 (Schedule): Second parent schedule.

        Yields:
            Iterator[Event]: Genes from each parents.

        """

    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents."""
        p1, p2 = parents
        p1_genes, p2_genes = self.get_genes(p1, p2)
        yield self._create_child(p1, p2, p1_genes, p2_genes)
        p2_genes, p1_genes = self.get_genes(p2, p1)
        yield self._create_child(p2, p1, p2_genes, p1_genes)


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

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for KPoint crossover."""
        p1_genes, p2_genes = [], []
        p1_extend = p1_genes.extend
        p2_extend = p2_genes.extend
        evts = self.events
        ne = len(evts)
        indices = sorted([0, *self.rng.sample(range(1, ne), k=self.k), ne])
        for i, events in enumerate(evts[i:j] for i, j in pairwise(indices)):
            if i % 2 == 0:
                p1_extend(events)
            else:
                p2_extend(events)
        return p1_genes, p2_genes


@dataclass(slots=True)
class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for Scattered crossover."""
        evts = self.events
        indices = self.rng.sample(range(len(evts)), k=len(evts))
        mid = len(evts) // 2
        return [evts[i] for i in indices[:mid]], [evts[i] for i in indices[mid:]]


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    The main difference with Scattered, is Scattered guarantees close to 50/50 splits.
    Uniform may result in more imbalanced splits.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for Uniform crossover."""
        p1_genes, p2_genes = [], []
        randint = self.rng.randint
        p1_append = p1_genes.append
        p2_append = p2_genes.append
        evts = self.events
        for e in evts:
            idx = randint(1, 2)
            if idx == 1:
                p1_append(e)
            elif idx == 2:
                p2_append(e)
        return p1_genes, p2_genes


@dataclass(slots=True)
class Partial(EventCrossover):
    """Partial crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a 3 sided coin for each gene.
    The third side is left unselected, leaving the repairer to fix the schedule.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for Partial crossover."""
        p1_genes, p2_genes = [], []
        randint = self.rng.randint
        p1_append = p1_genes.append
        p2_append = p2_genes.append
        evts = self.events
        for e in evts:
            idx = randint(1, 3)
            if idx == 1:
                p1_append(e)
            elif idx == 2:
                p2_append(e)
        return p1_genes, p2_genes


@dataclass(slots=True)
class RoundTypeCrossover(EventCrossover):
    """Round type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for RoundType crossover."""
        p1_genes, p2_genes = [], []
        p1_extend = p1_genes.extend
        p2_extend = p2_genes.extend
        evts_by_rt = self.event_factory.as_roundtypes()
        for i, evts in enumerate(evts_by_rt.values()):
            if i % 2 == 0:
                p1_extend(evts)
            else:
                p2_extend(evts)
        return p1_genes, p2_genes


@dataclass(slots=True)
class TimeSlotCrossover(EventCrossover):
    """Time slot crossover operator for genetic algorithms.

    Each gene is chosen based on the time slot of the event.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for TimeSlot crossover."""
        p1_genes, p2_genes = [], []
        p1_extend = p1_genes.extend
        p2_extend = p2_genes.extend
        evts_by_ts = self.event_factory.as_timeslots()
        for i, evts in enumerate(evts_by_ts.values()):
            if i % 2 == 0:
                p1_extend(evts)
            else:
                p2_extend(evts)
        return p1_genes, p2_genes


@dataclass(slots=True)
class LocationCrossover(EventCrossover):
    """Location crossover operator for genetic algorithms.

    Each gene is chosen based on the location of the event.
    """

    def get_genes(self) -> Iterator[Iterator[Event]]:
        """Get the genes for Location crossover."""
        p1_genes, p2_genes = [], []
        p1_extend = p1_genes.extend
        p2_extend = p2_genes.extend
        evts_by_loc = self.event_factory.as_locations()
        for i, evts in enumerate(evts_by_loc.values()):
            if i % 2 == 0:
                p1_extend(evts)
            else:
                p2_extend(evts)
        return p1_genes, p2_genes


@dataclass(slots=True)
class BestTeamCrossover(TeamCrossover):
    """Team crossover operator for genetic algorithms.

    This operator combines events from two parent schedules based on team assignments.
    """

    def get_genes(self, p1: Schedule, p2: Schedule) -> Iterator[Iterator[Event]]:
        """Get the genes for Team crossover."""
        evts_map = self.event_map
        p1_best = set()
        p2_best = set()
        p1_update = p1_best.update
        p2_update = p2_best.update
        p1_data = ((sum(t.fitness), t.events) for t in p1.all_teams())
        p2_data = ((sum(t.fitness), t.events) for t in p2.all_teams())
        for (t1_fit, t1_events), (t2_fit, t2_events) in zip(p1_data, p2_data, strict=True):
            if t1_fit > t2_fit:
                p1_update(t1_events)
            elif t2_fit > t1_fit:
                p2_update(t2_events)
        yield (evts_map[e] for e in p1_best)
        yield (evts_map[e] for e in p2_best.difference(p1_best))
