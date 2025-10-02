"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import CrossoverOp
from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.app_config import AppConfig
    from ..data_model.event import Event, EventFactory
    from ..data_model.team import TeamFactory

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
        CrossoverOp.ROUND_TYPE_CROSSOVER: lambda: RoundTypeCrossover(team_factory, event_factory, rng),
        CrossoverOp.TIMESLOT_CROSSOVER: lambda: TimeSlotCrossover(team_factory, event_factory, rng),
        CrossoverOp.LOCATION_CROSSOVER: lambda: LocationCrossover(team_factory, event_factory, rng),
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
    rng: np.random.Generator
    events: list[Event] = field(init=False, repr=False)
    n_evts: int = field(init=False, repr=False)
    indices: np.ndarray = None

    def __post_init__(self) -> None:
        """Post-initialization to validate the crossover operator."""
        self.events = self.event_factory.build_singles_or_side1()
        self.n_evts = len(self.events)
        self.indices = np.arange(self.n_evts)

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
        c = Schedule(teams=self.team_factory.build(), origin=f"(C | {self!s})")
        self.assign_from_p1(c, p1, p1_genes)
        self.assign_from_p2(c, p2, p2_genes)
        return c

    def assign_from_p1(self, c: Schedule, p: Schedule, p1_genes: Iterator[int]) -> None:
        """Assign genes."""
        for i in p1_genes:
            e1 = self.events[i]
            if (pt1 := p[e1]) is None:
                continue

            t1 = c.get_team(pt1.idx)
            if (e2 := e1.paired) is None:
                c.assign_single(e1, t1)
            elif (pt2 := p[e2]) is not None:
                t2 = c.get_team(pt2.idx)
                c.assign_match(e1, e2, t1, t2)

    def assign_from_p2(self, c: Schedule, p: Schedule, p2_genes: Iterator[int]) -> None:
        """Assign genes."""
        for i in p2_genes:
            e1 = self.events[i]
            if (pt1 := p[e1]) is None:
                continue

            t1 = c.get_team(pt1.idx)
            if not c.team_needs_round(t1, e1.roundtype) or c.conflicts(t1, e1):
                continue

            if (e2 := e1.paired) is None:
                c.assign_single(e1, t1)
            elif (pt2 := p[e2]) is not None:
                t2 = c.get_team(pt2.idx)
                if c.team_needs_round(t2, e2.roundtype) and not c.conflicts(t2, e2):
                    c.assign_match(e1, e2, t1, t2)


@dataclass(slots=True)
class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        if hasattr(self, "k"):
            return f"{self.__class__.__name__}(k={self.k})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self) -> Iterator[Iterator[int]]:
        """Get the genes for the crossover.

        Yields:
            Iterator[int]: Genes for each parents.

        """

    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents."""
        p1, p2 = parents
        p1_genes, p2_genes = self.get_genes()
        yield self._create_child(p1, p2, p1_genes, p2_genes)
        yield self._create_child(p2, p1, p2_genes, p1_genes)


@dataclass(slots=True)
class KPoint(EventCrossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = field(default=1)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(KPoint, self).__post_init__()
        if not 1 <= self.k < self.n_evts:
            msg = "k must be between 1 and the number of events."
            raise ValueError(msg)

    def get_genes(self) -> Iterator[Iterator[int]]:
        """Get the genes for KPoint crossover."""
        # Select k random split points from the middle of the indices.
        splits = self.rng.choice(self.indices[1:-1], size=self.k, replace=False)
        splits.sort()
        # Create segments and alternate between parents.
        segments = np.split(self.indices, splits)
        return np.concatenate(segments[::2]), np.concatenate(segments[1::2])


@dataclass(slots=True)
class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self) -> Iterator[Iterator[int]]:
        """Get the genes for Scattered crossover."""
        # Shuffle all indices and split in half.
        permuted_indices = self.rng.permutation(self.indices)
        return np.array_split(permuted_indices, 2)


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    The main difference with Scattered, is Scattered guarantees close to 50/50 splits.
    Uniform may result in more imbalanced splits.
    """

    def get_genes(self) -> Iterator[Iterator[int]]:
        """Get the genes for Uniform crossover."""
        # Create a mask for selecting genes from each parent.
        mask = self.rng.integers(0, 2, size=self.n_evts, dtype=bool)
        return self.indices[mask], self.indices[~mask]


@dataclass(slots=True)
class StructureCrossover(EventCrossover):
    """Structure-based crossover operator for genetic algorithms.

    Each gene is chosen based on a specific structure of the event.
    """

    array: np.ndarray = None
    mid: int = None

    def get_genes(self) -> Iterator[Iterator[int]]:
        """Get the genes for Structure-based crossover."""
        self.rng.shuffle(self.indices)
        p1_indices = self.array[self.indices[: self.mid]]
        p2_indices = self.array[self.indices[self.mid :]]
        return p1_indices[p1_indices >= 0], p2_indices[p2_indices >= 0]


@dataclass(slots=True)
class RoundTypeCrossover(StructureCrossover):
    """Round type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(RoundTypeCrossover, self).__post_init__()
        roundtypes = list(self.event_factory.as_roundtypes().keys())
        roundtype_event_map = {rt: [] for rt in roundtypes}
        for i, event in enumerate(self.events):
            roundtype_event_map[event.roundtype].append(i)

        self.array = np.full((len(roundtypes), max(len(evts) for evts in roundtype_event_map.values())), -1, dtype=int)
        for j, rt in enumerate(roundtypes):
            evts = roundtype_event_map[rt]
            self.array[j, : len(evts)] = evts

        self.indices = np.array(range(len(roundtypes)))
        self.mid = len(self.indices) // 2


@dataclass(slots=True)
class TimeSlotCrossover(StructureCrossover):
    """Time slot crossover operator for genetic algorithms.

    Each gene is chosen based on the time slot of the event.
    """

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(TimeSlotCrossover, self).__post_init__()
        timeslot_ids = np.array(list({t[1].idx for t in self.event_factory.as_timeslots()}))
        timeslot_event_map = {ts_id: [] for ts_id in timeslot_ids}
        for i, event in enumerate(self.events):
            timeslot_event_map[event.timeslot.idx].append(i)

        self.array = np.full((len(timeslot_ids), max(len(evts) for evts in timeslot_event_map.values())), -1, dtype=int)
        for j, ts_id in enumerate(timeslot_ids):
            evts = timeslot_event_map[ts_id]
            self.array[j, : len(evts)] = evts

        self.indices = np.array(range(len(timeslot_ids)))
        self.mid = len(self.indices) // 2


@dataclass(slots=True)
class LocationCrossover(StructureCrossover):
    """Location crossover operator for genetic algorithms.

    Each gene is chosen based on the location of the event.
    """

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super(LocationCrossover, self).__post_init__()
        location_ids = np.array(list({loc[1].idx for loc in self.event_factory.as_locations()}))
        location_event_map = {loc_id: [] for loc_id in location_ids}
        for i, event in enumerate(self.events):
            location_event_map[event.location.idx].append(i)

        self.array = np.full((len(location_ids), max(len(evts) for evts in location_event_map.values())), -1, dtype=int)
        for j, loc_id in enumerate(location_ids):
            evts = location_event_map[loc_id]
            self.array[j, : len(evts)] = evts

        self.indices = np.array(range(len(location_ids)))
        self.mid = len(self.indices) // 2
