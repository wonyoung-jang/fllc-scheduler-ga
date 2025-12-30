"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import CrossoverOp
from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ..config.pydantic_schemas import OperatorConfig
    from ..data_model.event import EventFactory, EventProperties

logger = getLogger(__name__)


def build_crossovers(
    rng: np.random.Generator,
    operators: OperatorConfig,
    event_factory: EventFactory,
    event_properties: EventProperties,
) -> tuple[Crossover, ...]:
    """Build and return a tuple of crossover operators based on the configuration."""
    if not (crossover_types := operators.crossover.types):
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return ()

    crossover_factory = {
        CrossoverOp.K_POINT: lambda p, k: KPoint(**p, k=k),
        CrossoverOp.SCATTERED: Scattered,
        CrossoverOp.UNIFORM: Uniform,
        CrossoverOp.ROUND_TYPE_CROSSOVER: RoundTypeCrossover,
        CrossoverOp.TIMESLOT_CROSSOVER: TimeSlotCrossover,
        CrossoverOp.LOCATION_CROSSOVER: LocationCrossover,
    }
    params = {
        "event_factory": event_factory,
        "event_properties": event_properties,
        "rng": rng,
    }

    def _generate_crossovers() -> Iterator[Crossover]:
        for crossover_name in crossover_types:
            if crossover_name not in crossover_factory:
                msg = f"Unknown crossover type in config: {crossover_name}"
                raise ValueError(msg)

            if crossover_name == CrossoverOp.K_POINT:
                if crossover_ks := operators.crossover.k_vals:
                    for k in crossover_ks:
                        if k <= 0:
                            msg = f"Invalid crossover k value: {k}. Must be greater than 0."
                            raise ValueError(msg)

                        yield crossover_factory[crossover_name](params, k)
            else:
                yield crossover_factory[crossover_name](**params)

    return tuple(_generate_crossovers())


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    event_factory: EventFactory
    event_properties: EventProperties
    rng: np.random.Generator

    events: np.ndarray = field(init=False)
    n_evts: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the crossover operator."""
        self.events = self.event_factory.build_singles_or_side1_indices()
        self.n_evts = self.events.shape[0]

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
        p1: np.ndarray,
        p2: np.ndarray,
        p1_genes: np.ndarray,
        p2_genes: np.ndarray,
    ) -> Schedule:
        """Create a child schedule from two parents."""
        child = Schedule(origin=f"(C | {self!s})")
        self.assign_from_p1(child, p1, p1_genes)
        self.assign_from_p2(child, p2, p2_genes)
        return child

    def assign_from_p1(self, child: Schedule, p1: np.ndarray, p1_genes: np.ndarray) -> None:
        """Assign genes."""
        p1_gene_pairs = self.event_properties.paired_idx[p1_genes]
        for e1, e2 in zip(p1_genes, p1_gene_pairs, strict=True):
            t1 = p1[e1]
            if e2 == -1:
                child.assign(t1, e1)
            else:
                t2 = p1[e2]
                child.assign(t1, e1)
                child.assign(t2, e2)

    def assign_from_p2(self, child: Schedule, p2: np.ndarray, p2_genes: np.ndarray) -> None:
        """Assign genes."""
        p2_genes_pairs = self.event_properties.paired_idx[p2_genes]
        p2_genes_rt = self.event_properties.roundtype_idx[p2_genes]
        for e1, e2, rt in zip(p2_genes, p2_genes_pairs, p2_genes_rt, strict=True):
            t1 = p2[e1]
            if t1 == -1 or not child.needs_round(t1, rt) or child.conflicts(t1, e1):
                continue

            if e2 == -1:
                child.assign(t1, e1)
            else:
                t2 = p2[e2]
                if t2 == -1 or not child.needs_round(t2, rt) or child.conflicts(t2, e2):
                    continue

                child.assign(t1, e1)
                child.assign(t2, e2)


@dataclass(slots=True)
class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        if (k := getattr(self, "k", None)) is not None:
            return f"{self.__class__.__name__}(k={k})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for the crossover.

        Returns:
            Iterable[np.ndarray]: Genes for each parent.

        """

    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents."""
        i, j = parents
        p1 = i.schedule
        p2 = j.schedule
        p1_genes, p2_genes = self.get_genes()
        yield self._create_child(p1, p2, p1_genes, p2_genes)
        yield self._create_child(p2, p1, p2_genes, p1_genes)


@dataclass(slots=True)
class KPoint(EventCrossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = 1

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super().__post_init__()
        if not 1 <= self.k < self.n_evts:
            logger.warning("Invalid k value for KPoint crossover: %d. Setting k to 1.", self.k)
            self.k = 1

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for KPoint crossover."""
        n = self.n_evts

        # Single-point crossover
        if self.k == 1:
            # Pick a split point index directly (1 to n-1)
            split = self.rng.integers(1, n)
            return self.events[:split], self.events[split:]

        # Multi-point crossover
        splits = self.rng.choice(n - 1, size=self.k, replace=False) + 1
        mask = np.zeros(n, dtype=bool)
        mask[splits] = True
        np.bitwise_xor.accumulate(mask, out=mask)
        return self.events[mask], self.events[~mask]


@dataclass(slots=True)
class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Scattered crossover."""
        # Shuffle all indices and split in half.
        permuted_indices = self.rng.permutation(self.events)
        return np.array_split(permuted_indices, 2)


@dataclass(slots=True)
class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    The main difference with Scattered, is Scattered guarantees close to 50/50 splits.
    Uniform may result in more imbalanced splits.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Uniform crossover."""
        # Create a mask for selecting genes from each parent.
        mask = self.rng.random(self.n_evts) < 0.5
        return self.events[mask], self.events[~mask]


@dataclass(slots=True)
class StructureCrossover(EventCrossover):
    """Structure-based crossover operator for genetic algorithms.

    Each gene is chosen based on a specific structure of the event.
    """

    lookup: np.ndarray = field(init=False)
    structure: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super().__post_init__()
        self._initialize_attributes()

    @abstractmethod
    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""

    def _initialize_attributes(self) -> None:
        """Initialize attributes specific to the structure crossover."""
        eventmap = defaultdict(list)

        keys = self._get_group_keys()
        for key, e in zip(keys, self.events, strict=True):
            eventmap[key].append(e)

        sorted_keys = sorted(eventmap.keys())
        unique_ids = np.array(sorted_keys)
        n_ids = unique_ids.shape[0]
        max_len = max(len(evts) for evts in eventmap.values())

        self.lookup = np.full((n_ids, max_len), -1, dtype=int)

        for i, uid in enumerate(unique_ids):
            evts = eventmap[uid]
            self.lookup[i, : len(evts)] = evts

        self.structure = np.arange(n_ids)

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Structure-based crossover."""
        self.rng.shuffle(self.structure)
        p1, p2 = np.array_split(self.structure, indices_or_sections=2, axis=0)
        p1_indices = self.lookup[p1]
        p2_indices = self.lookup[p2]
        return p1_indices[p1_indices >= 0], p2_indices[p2_indices >= 0]


@dataclass(slots=True)
class RoundTypeCrossover(StructureCrossover):
    """TournamentRound type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.event_properties.roundtype_idx[self.events]


@dataclass(slots=True)
class TimeSlotCrossover(StructureCrossover):
    """Time slot crossover operator for genetic algorithms.

    Each gene is chosen based on the time slot of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.event_properties.timeslot_idx[self.events]


@dataclass(slots=True)
class LocationCrossover(StructureCrossover):
    """Location crossover operator for genetic algorithms.

    Each gene is chosen based on the location of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.event_properties.loc_idx[self.events]
