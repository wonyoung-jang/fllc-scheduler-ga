"""Genetic operators."""

import itertools
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import EPSILON, CrossoverOp, MutationOp, SelectionOp
from fll_scheduler_ga.domain.model import Schedule

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TournamentConfig

logger = logging.getLogger(__name__)

########################################################################
###  NSGA-III
########################################################################


def calc_ref_points(n_obj: int, n_pop: int) -> np.ndarray:
    """Generate a set of structured reference points."""
    p = 1
    while math.comb(n_obj + p - 1, n_obj - 1) < n_pop:
        p += 1

    def _gen() -> Iterator[np.ndarray]:
        for dividers in itertools.combinations(range(n_obj + p - 1), n_obj - 1):
            coords = np.zeros(n_obj, dtype=float)
            prev = -1
            for i, divider in enumerate(dividers):
                coords[i] = divider - prev - 1
                prev = divider
            coords[-1] = n_obj + p - 1 - dividers[-1] - 1
            yield (coords / p)

    points = np.array(list(_gen()), dtype=float)
    logger.debug("Generated %d reference points:\n%s", points.shape[0], points)
    return points


def calc_norm_sq_of_refs(points: np.ndarray) -> np.ndarray:
    """Calculate the squared norms of the reference points."""
    norm_sq = (points * points).sum(axis=1)
    norm_sq[norm_sq == 0.0] = EPSILON
    logger.debug("Computed squared norms of reference points:\n%s", norm_sq)
    return norm_sq


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: np.random.Generator
    n_refs: int
    points: np.ndarray
    norm_sq: np.ndarray

    def get_fronts(self, fits: np.ndarray, n_pop: int) -> list[np.ndarray]:
        """Perform non-dominated sorting on the population."""
        n_fit = fits.shape[0]
        if n_fit == 0:
            return []
        # dom[i,j] = True if i dominates j (>= on all and > on at least one)
        dom = np.logical_and(
            (fits[:, None, :] >= fits[None, :, :]).all(axis=2),
            (fits[:, None, :] > fits[None, :, :]).any(axis=2),
        )
        dom_count = dom.sum(axis=0)
        assigned = np.zeros(n_fit, dtype=bool)
        fronts: list[np.ndarray] = []
        current_front: np.ndarray = (dom_count == 0).nonzero()[0]
        assigned[current_front] = True
        fronts.append(current_front)
        n_ranked = current_front.size
        while n_ranked < n_pop and current_front.size > 0:
            dom_count = dom_count - dom[current_front, :].sum(axis=0)
            next_front: np.ndarray = ((dom_count == 0) & (~assigned)).nonzero()[0]
            if next_front.size == 0:
                break
            assigned[next_front] = True
            fronts.append(next_front)
            n_ranked += next_front.size
            current_front = next_front
        return fronts

    def norm_and_associate(self, fits: np.ndarray) -> tuple[np.ndarray, ...]:
        """Normalize objectives then associate individuals with nearest reference points."""
        ideal = fits.max(axis=0)
        nadir = fits.min(axis=0)
        span = ideal - nadir
        span[span == 0.0] = EPSILON
        norm = (ideal - fits) / span
        coeffs = (norm @ self.points.T) / self.norm_sq
        coeffs[coeffs < 0.0] = 0.0
        proj = coeffs[:, :, None] * self.points[None, :, :]
        residual = norm[:, None, :] - proj
        dist: np.ndarray = np.linalg.norm(residual, axis=2)
        min_dist = dist.min(axis=1)
        tie = dist == min_dist[:, None]
        rand_matrix = self.rng.random(dist.shape)
        rand_matrix[~tie] = -1.0
        return rand_matrix.argmax(axis=1), min_dist

    def ranks_from_fronts(self, fronts: list[np.ndarray], n_individuals: int) -> np.ndarray:
        """Assign ranks to individuals based on their fronts."""
        rank = np.full(n_individuals, fill_value=-1, dtype=int)
        for r, front in enumerate(fronts):
            rank[front] = r
        return rank

    def niche(
        self, count: np.ndarray, n_last: int, n_remaining: int, niche_ref: np.ndarray, niche_dist: np.ndarray
    ) -> np.ndarray:
        """Select k individuals from the last front using a niching mechanism."""
        mask = np.ones(n_last, dtype=bool)
        n_selected = 0
        while n_selected < n_remaining:
            available_ref = np.unique(niche_ref[mask])
            ref_count = count[available_ref]
            niche_idx = available_ref[(ref_count == ref_count.min()).nonzero()[0]]
            niche_idx = niche_idx[self.rng.permutation(niche_idx.size)[: n_remaining - n_selected]]
            for niche_i in niche_idx:
                next_i = ((niche_ref == niche_i) & mask).nonzero()[0]
                self.rng.shuffle(next_i)
                index = next_i[niche_dist[next_i].argmin()] if count[niche_i] == 0 else next_i[0]
                mask[index] = False
                count[niche_i] += 1
                n_selected += 1
                if n_selected >= n_remaining:
                    break
        return (~mask).nonzero()[0]

    def count(self, niche_selected: np.ndarray) -> np.ndarray:
        """Count how many individuals are associated with each reference point."""
        idx, c = np.unique(niche_selected, return_counts=True)
        count = np.zeros(self.n_refs, dtype=int)
        count[idx] = c
        return count

    def _select_result(
        self, fronts: list[np.ndarray], fits: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        flat = np.concatenate(fronts)
        ranks = self.ranks_from_fronts(fronts, fits.shape[0])
        return fronts, flat, ranks[flat]

    def select(self, fits: np.ndarray, n_pop: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Select the next generation using NSGA-III principles."""
        fronts = self.get_fronts(fits, n_pop)
        last_i = len(fronts) - 1
        select_idx = np.array([i for f in fronts for i in f], dtype=int)
        if len(fronts) == 1:
            fronts[0] = self.rng.permutation(select_idx)[:n_pop]
            return self._select_result(fronts, fits)
        ref, dist = self.norm_and_associate(fits[select_idx])
        n_last = fronts[last_i].size
        niche = self.niche(
            count=self.count(ref[:-n_last]),
            n_last=n_last,
            n_remaining=n_pop - select_idx[:-n_last].size,
            niche_ref=ref[-n_last:],
            niche_dist=dist[-n_last:],
        )
        fronts[last_i] = fronts[last_i][niche]
        return self._select_result(fronts, fits)


########################################################################
###  Selection
########################################################################


class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    @abstractmethod
    def select(self, n: int) -> np.ndarray: ...


@dataclass(slots=True)
class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    rng: np.random.Generator

    def __str__(self) -> str:
        """Return a string representation of the selection operator."""
        return SelectionOp.RANDOM_SELECT

    def select(self, n: int) -> np.ndarray:
        """Select individuals from the population to form the next generation."""
        i1 = self.rng.integers(0, n)
        i2 = self.rng.integers(0, n)
        while i1 == i2:
            i2 = self.rng.integers(0, n)
        return np.array((i1, i2), dtype=int)


########################################################################
###  Crossover
########################################################################
def build_crossovers(
    rng: np.random.Generator,
    types: tuple[str, ...],
    crossover_ks: tuple[int, ...],
    evt_repo: EventRepository,
    evt_prop: EventProperties,
) -> tuple[Crossover, ...]:
    """Build and return a tuple of crossover operators based on the configuration."""
    factory: dict[str, Callable] = {
        CrossoverOp.K_POINT: KPoint,
        CrossoverOp.SCATTERED: Scattered,
        CrossoverOp.UNIFORM: Uniform,
        CrossoverOp.ROUND_TYPE_CROSSOVER: RoundTypeCrossover,
        CrossoverOp.TIMESLOT_CROSSOVER: TimeSlotCrossover,
        CrossoverOp.LOCATION_CROSSOVER: LocationCrossover,
    }
    p = {
        "evt": evt_repo.singles_or_side1_idx,
        "n_evt": evt_repo.singles_or_side1_idx.shape[0],
        "evt_prop": evt_prop,
        "rng": rng,
    }
    crossover_ks = tuple(k for k in crossover_ks if 1 <= k < p["n_evt"])

    def _gen() -> Iterator[Crossover]:
        for name in types:
            if name == CrossoverOp.K_POINT:
                yield from (factory[name](**p, k=k) for k in crossover_ks)
            else:
                yield factory[name](**p)

    return tuple(_gen())


@dataclass(slots=True)
class Crossover(ABC):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    evt_prop: EventProperties
    rng: np.random.Generator
    evt: np.ndarray
    n_evt: int

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        if (k := getattr(self, "k", None)) is not None:
            return f"{self.__class__.__name__}(k={k})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self) -> Iterable[np.ndarray]: ...

    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Produce child schedules from two parents."""
        _p1, _p2 = parents
        p1 = _p1.schedule
        p2 = _p2.schedule
        p1_genes, p2_genes = self.get_genes()
        yield self._create_child(p1, p2, p1_genes, p2_genes)
        yield self._create_child(p2, p1, p2_genes, p1_genes)

    def _create_child(self, p1: np.ndarray, p2: np.ndarray, p1_genes: np.ndarray, p2_genes: np.ndarray) -> Schedule:
        """Create a child schedule from two parents."""
        child = Schedule(origin=f"(C | {self!s})")
        self._assign_from_p1(child, p1, p1_genes)
        self._assign_from_p2(child, p2, p2_genes)
        return child

    def _assign_from_p1(self, child: Schedule, p1: np.ndarray, p1_genes: np.ndarray) -> None:
        """Assign genes."""
        for e1, e2 in zip(p1_genes, self.evt_prop.paired_idx[p1_genes], strict=True):
            if e2 == -1:
                child.assign(p1[e1], e1)
            else:
                child.assign(p1[e1], e1)
                child.assign(p1[e2], e2)

    def _assign_from_p2(self, child: Schedule, p2: np.ndarray, p2_genes: np.ndarray) -> None:
        """Assign genes."""
        for e1, e2, rt in zip(
            p2_genes, self.evt_prop.paired_idx[p2_genes], self.evt_prop.roundtype_idx[p2_genes], strict=True
        ):
            t1 = p2[e1]
            if t1 != -1 and child.needs_round(t1, rt) and not child.conflicts(t1, e1):
                if e2 == -1:
                    child.assign(t1, e1)
                else:
                    t2 = p2[e2]
                    if t2 != -1 and child.needs_round(t2, rt) and not child.conflicts(t2, e2):
                        child.assign(t1, e1)
                        child.assign(t2, e2)


@dataclass(slots=True)
class KPoint(Crossover):
    """K-point crossover operator for genetic algorithms."""

    k: int = 1

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for KPoint crossover."""
        # Single-point crossover
        if self.k == 1:
            split = self.rng.integers(1, self.n_evt)
            return self.evt[:split], self.evt[split:]
        # Multi-point crossover
        splits = self.rng.choice(self.n_evt - 1, size=self.k, replace=False) + 1
        mask = np.zeros(self.n_evt, dtype=bool)
        mask[splits] = True
        np.bitwise_xor.accumulate(mask, out=mask)
        return self.evt[mask], self.evt[~mask]


class Scattered(Crossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Scattered crossover."""
        return np.array_split(self.rng.permutation(self.evt), 2)


class Uniform(Crossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    The main difference with Scattered, is Scattered guarantees close to 50/50 splits.
    Uniform may result in more imbalanced splits.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Uniform crossover."""
        mask = self.rng.random(self.n_evt) < 0.5
        return self.evt[mask], self.evt[~mask]


@dataclass(slots=True)
class StructureCrossover(Crossover):
    """Structure-based crossover operator for genetic algorithms.

    Each gene is chosen based on a specific structure of the event.
    """

    _lookup: np.ndarray = field(init=False)
    _structure: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        emap = defaultdict(list)
        for key, e in zip(self._groupkey(), self.evt, strict=True):
            emap[key].append(e)

        max_len = max(len(e) for e in emap.values())
        self._lookup = np.full((len(emap), max_len), -1, dtype=int)
        for i, uid in enumerate(emap):
            evt = emap[uid]
            self._lookup[i, : len(evt)] = evt

        self._structure = np.arange(len(emap))

    @abstractmethod
    def _groupkey(self) -> np.ndarray: ...

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Structure-based crossover."""
        self.rng.shuffle(self._structure)
        p1, p2 = np.array_split(self._structure, indices_or_sections=2, axis=0)
        p1_idx = self._lookup[p1]
        p2_idx = self._lookup[p2]
        return p1_idx[p1_idx >= 0], p2_idx[p2_idx >= 0]


class RoundTypeCrossover(StructureCrossover):
    """Roundtype crossover operator for genetic algorithms. Each gene is chosen based on the round type of the event."""

    def _groupkey(self) -> np.ndarray:
        return self.evt_prop.roundtype_idx[self.evt]


class TimeSlotCrossover(StructureCrossover):
    """Time slot crossover operator for genetic algorithms. Each gene is chosen based on the time slot of the event."""

    def _groupkey(self) -> np.ndarray:
        return self.evt_prop.timeslot_idx[self.evt]


class LocationCrossover(StructureCrossover):
    """Location crossover operator for genetic algorithms. Each gene is chosen based on the location of the event."""

    def _groupkey(self) -> np.ndarray:
        return self.evt_prop.loc_idx[self.evt]


########################################################################
###  Mutation
########################################################################
type Match = tuple[int, int, int, int]


def build_mutations(
    rng: np.random.Generator, types: tuple[str, ...], evt_repo: EventRepository, evt_prop: EventProperties
) -> tuple[Mutation, ...]:
    """Build and return a tuple of mutation operators based on the configuration."""
    factory: dict[str, Callable[[dict], Mutation]] = {
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda p: SwapMatchMutation(**p),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda p: SwapMatchMutation(**p, same_location=True),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda p: SwapMatchMutation(**p, same_timeslot=True),
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda p: SwapTeamMutation(**p),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda p: SwapTeamMutation(**p, same_location=True),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda p: SwapTeamMutation(**p, same_timeslot=True),
        MutationOp.SWAP_TABLE_SIDE: lambda p: SwapTableSideMutation(**p, same_timeslot=True, same_location=True),
        MutationOp.INVERSION: lambda p: InversionMutation(**p),
        MutationOp.SCRAMBLE: lambda p: ScrambleMutation(**p),
    }
    params = {"rng": rng, "evt_repo": evt_repo, "evt_prop": evt_prop}
    return tuple(factory[name](params) for name in types)


@dataclass(slots=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: np.random.Generator
    evt_repo: EventRepository
    evt_prop: EventProperties

    @abstractmethod
    def mutate(self, s: Schedule) -> bool: ...


@dataclass(slots=True)
class SwapMutation(Mutation):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    same_timeslot: bool = False
    same_location: bool = False
    _candidates: list[tuple[tuple[int, ...], ...]] = field(default_factory=list)
    _n_candidates: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        for matches in self.evt_repo.matches.values():
            for (e1a, e1b), (e2a, e2b) in itertools.combinations(matches, 2):
                ts_ok = (self.evt_prop.timeslot_idx[e1a] == self.evt_prop.timeslot_idx[e2a]) == self.same_timeslot
                loc_ok = (self.evt_prop.loc_idx[e1a] == self.evt_prop.loc_idx[e2a]) == self.same_location
                if (ts_ok and loc_ok) or (self.same_timeslot and self.same_location):
                    self._candidates.append(((e1a, e1b), (e2a, e2b)))
        self._n_candidates = len(self._candidates)
        logger.debug("Initialized %d swap candidates for %s", self._n_candidates, str(self))

    @abstractmethod
    def get_swap_candidates(self, s: Schedule) -> tuple[Match, ...] | tuple[None, ...]: ...


class SwapTeamMutation(SwapMutation):
    """Mutation operator for swapping single team between two matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_TEAM_SAME_LOCATION,
            (False, True): MutationOp.SWAP_TEAM_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

    def mutate(self, s: Schedule) -> bool:
        """Swap one team from two different matches."""
        if self._n_candidates <= 0:
            return False
        match1, match2 = self.get_swap_candidates(s)
        if match1 is None or match2 is None:
            return False
        e1a, _, t1a, _ = match1
        e2a, _, t2a, _ = match2
        s.swap_assignment(t1a, e1a, e2a)
        s.swap_assignment(t2a, e2a, e1a)
        return True

    def get_swap_candidates(self, s: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        for i in self.rng.permutation(self._n_candidates):
            (e1a, e1b), (e2a, e2b) = self._candidates[i]
            t1a, t1b = s.schedule[e1a], s.schedule[e1b]
            t2a, t2b = s.schedule[e2a], s.schedule[e2b]
            match_team_ids = {t1a, t1b, t2a, t2b}
            if len(match_team_ids) < 4 or s.conflicts(t1a, e2a, ignore=e1a) or s.conflicts(t2a, e1a, ignore=e2a):
                continue
            return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)
        return None, None


class SwapMatchMutation(SwapMutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_MATCH_SAME_LOCATION,
            (False, True): MutationOp.SWAP_MATCH_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

    def mutate(self, s: Schedule) -> bool:
        """Swap two entire matches."""
        if self._n_candidates <= 0:
            return False
        match1, match2 = self.get_swap_candidates(s)
        if match1 is None or match2 is None:
            return False
        e1a, e1b, t1a, t1b = match1
        e2a, e2b, t2a, t2b = match2
        if -1 not in (t1a, t1b):
            s.swap_assignment(t1a, e1a, e2a)
            s.swap_assignment(t1b, e1b, e2b)
        if -1 not in (t2a, t2b):
            s.swap_assignment(t2a, e2a, e1a)
            s.swap_assignment(t2b, e2b, e1b)
        return True

    def get_swap_candidates(self, s: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        for i in self.rng.permutation(self._n_candidates):
            (e1a, e1b), (e2a, e2b) = self._candidates[i]
            t1a, t1b = s.schedule[e1a], s.schedule[e1b]
            if -1 not in (t1a, t1b) and (s.conflicts(t1a, e2a, ignore=e1a) or s.conflicts(t1b, e2b, ignore=e1b)):
                continue
            t2a, t2b = s.schedule[e2a], s.schedule[e2b]
            if -1 not in (t2a, t2b) and (s.conflicts(t2a, e1a, ignore=e2a) or s.conflicts(t2b, e1b, ignore=e2b)):
                continue
            return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)
        return None, None


class SwapTableSideMutation(SwapMutation):
    """Mutation operator for swapping the sides of two tables in a match."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SWAP_TABLE_SIDE

    def mutate(self, s: Schedule) -> bool:
        """Swap the sides of two tables in a match."""
        if self._n_candidates <= 0:
            return False
        match, _ = self.get_swap_candidates(s)
        if match is None:
            return False
        e1a, e1b, t1a, t1b = match
        s.swap_assignment(t1a, e1a, e1b)
        s.swap_assignment(t1b, e1b, e1a)
        return True

    def get_swap_candidates(self, s: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get one match to swap sides in the schedule schedule."""
        (e1a, e1b), _ = self._candidates[self.rng.integers(0, self._n_candidates)]
        return (e1a, e1b, s.schedule[e1a], s.schedule[e1b]), (-1, -1, -1, -1)


@dataclass(slots=True)
class TimeSlotSequenceMutation(Mutation):
    """Abstract base class for mutations that permute assignments within a single timeslot."""

    _candidate: dict[tuple[int, int], list[tuple[int, ...]]] = field(default_factory=dict)
    _key_to_tpr: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        for k, evt in self.evt_repo.timeslots.items():
            self._candidate[k] = [
                (e, self.evt_prop.paired_idx[e])
                for e in evt
                if self.evt_prop.loc_side[e] == 1 or self.evt_prop.paired_idx[e] == -1
            ]
            self._key_to_tpr[k] = self.evt_prop.teams_per_round[evt[0]]

    @abstractmethod
    def _permute_singles(self, items: list[int]) -> Iterator[int]: ...

    @abstractmethod
    def _permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]: ...

    def mutate(self, s: Schedule) -> bool:
        """Find a suitable timeslot and round type, then permute assignments."""
        i = self.rng.permutation(len(self._candidate))
        k = list(self._candidate.keys())[i[0]]
        if self._key_to_tpr[k] == 1:
            self.mutate_singles(s, self._candidate[k])
        elif self._key_to_tpr[k] == 2:
            self.mutate_matches(s, self._candidate[k])
        return True

    def mutate_singles(self, s: Schedule, matches: list[tuple[int, ...]]) -> None:
        """Permute team assignments for single-team events."""
        old_i = [s.schedule[e] for e, _ in matches]
        new_i = self._permute_singles(old_i)
        for (e, _), ot, nt in zip(matches, old_i, new_i, strict=True):
            if ot != nt:
                s.unassign(ot, e)
                s.assign(nt, e)

    def mutate_matches(self, s: Schedule, matches: list[tuple[int, ...]]) -> None:
        """Permute team assignments for match-based events."""
        old_i: list[tuple[int, ...]] = [(s.schedule[e1], s.schedule[e2]) for e1, e2 in matches]
        new_i = self._permute_matches(old_i)
        for (e1, e2), (ot1, ot2), (nt1, nt2) in zip(matches, old_i, new_i, strict=True):
            if (ot1, ot2) != (nt1, nt2):
                s.unassign(ot1, e1)
                s.unassign(ot2, e2)
                s.assign(nt1, e1)
                s.assign(nt2, e2)


class InversionMutation(TimeSlotSequenceMutation):
    """Inverts a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.INVERSION

    def _permute_singles(self, items: list[int]) -> Iterator[int]:
        """Invert a random sub-sequence of the items."""
        return iter(items[::-1])

    def _permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Invert a random sub-sequence of the items."""
        return (tuple(reversed(pair)) for pair in reversed(items))


class ScrambleMutation(TimeSlotSequenceMutation):
    """Scrambles a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SCRAMBLE

    def _permute_singles(self, items: list[int]) -> Iterator[int]:
        """Scramble a random sub-sequence of the items."""
        return iter(self.rng.permutation(items))

    def _permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Scramble a random sub-sequence of the items."""
        order = self.rng.permutation(len(items))
        return (items[i] for i in order)


########################################################################
###  Repair
########################################################################


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    config: TournamentConfig
    evt_prop: EventProperties
    rng: np.random.Generator
    _rt_to_tpr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._rt_to_tpr = np.array(list(self.config.round_idx_to_tpr.values()), dtype=int)

    def repair(self, s: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.
        """
        if s.get_size() == self.config.total_slots_required:
            return True
        teams, events = self.get_rt_tpr_maps(s)
        return self.iterative_repair(s, teams, events)

    def iterative_repair(
        self, s: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        while s.get_size() < self.config.total_slots_required:
            if self._attempt_repair_step(s, teams, events):
                return True
            self._unassign_and_requeue_event(s, teams, events)
        return s.get_size() == self.config.total_slots_required

    def _attempt_repair_step(
        self, s: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> bool:
        """Attempt to apply a repair function for the current round type.

        Returns True if the schedule is considered resolved for this step.
        """
        for key, teams_for_rt in teams.items():
            _, tpr = key
            if not events.get(key):
                return True
            if tpr == 1:
                _teams, _events = self.repair_singles(dict(enumerate(teams_for_rt)), dict(enumerate(events[key])), s)
            elif tpr == 2:
                _teams, _events = self.repair_matches(dict(enumerate(teams_for_rt)), dict(enumerate(events[key])), s)
            teams[key] = _teams
            events[key] = _events
            if _teams:
                return False
        return True

    def _unassign_and_requeue_event(
        self, s: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> None:
        """Select a random scheduled event, handle pairing logic, and move it back to the queue."""
        event_indices = s.scheduled_events()
        self.rng.shuffle(event_indices)
        primary_event = event_indices[0]
        e_rt_idx = self.evt_prop.roundtype_idx[primary_event]
        ek = (e_rt_idx, self.config.round_idx_to_tpr[e_rt_idx])
        paired_event = self.evt_prop.paired_idx[primary_event]
        loc_side = self.evt_prop.loc_side[primary_event]
        if paired_event != -1:
            e1, e2 = (paired_event, primary_event) if loc_side == 2 else (primary_event, paired_event)
        else:
            e1, e2 = primary_event, None
        t1 = s.schedule[e1]
        events[ek].append(e1)
        teams[ek].append(t1)
        s.unassign(t1, e1)
        if e2 is not None and (t2 := s.schedule[e2]) != -1:
            teams[ek].append(t2)
            s.unassign(t2, e2)

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
        """Get the round type to team/player maps for the current schedule."""
        # 1. Team Map
        teams: dict[tuple[int, int], list[int]] = defaultdict(list)
        # Find (team_id, roundtype_id) where rounds are needed (>0)
        # team_rounds is shape (n_teams, n_round_types)
        t_idxs, rt_idxs = (schedule.team_rounds > 0).nonzero()
        if t_idxs.size > 0:
            # Get the counts (how many rounds needed)
            counts = schedule.team_rounds[t_idxs, rt_idxs]
            # If a team needs 2 rounds, we need 2 entries
            t_repeated = t_idxs.repeat(repeats=counts)  # ty:ignore[no-matching-overload]
            rt_repeated = rt_idxs.repeat(repeats=counts)  # ty:ignore[no-matching-overload]
            # Map roundtype to teams_per_round
            tpr_repeated = self._rt_to_tpr[rt_repeated]
            # Grouping by (rt, tpr)
            for i in range(len(t_repeated)):
                k = (rt_repeated[i], tpr_repeated[i])
                teams[k].append(t_repeated[i])
        # 2. Event Map
        events: dict[tuple[int, int], list[int]] = defaultdict(list)
        unscheduled = schedule.unscheduled_events()
        if unscheduled.size > 0:
            # Filter logic: (paired != -1 and side == 1) OR (paired == -1)
            paired = self.evt_prop.paired_idx[unscheduled]
            sides = self.evt_prop.loc_side[unscheduled]
            # Mask for valid repair candidates (singles or side 1 of matches)
            mask = (paired == -1) | (sides == 1)
            valid_events = unscheduled[mask]
            if valid_events.size > 0:
                valid_rts = self.evt_prop.roundtype_idx[valid_events]
                valid_tprs = self._rt_to_tpr[valid_rts]
                for i in range(len(valid_events)):
                    k = (valid_rts[i], valid_tprs[i])
                    if k in teams:
                        events[k].append(valid_events[i])
        return teams, events

    def repair_singles(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign single-team events to teams that need them."""
        while teams:
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tk = team_keys[0]
            t = teams.pop(tk)
            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)
            for ek in event_keys:
                e = events[ek]
                if not schedule.conflicts(t, e):
                    schedule.assign(t, e)
                    events.pop(ek)
                    break
            else:
                teams[tk] = t
                break
        return list(teams.values()), list(events.values())

    def repair_matches(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign match events to teams that need them."""
        while len(teams) >= 2:
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tk = team_keys[0]
            t1 = teams.pop(tk)
            for i, t2 in teams.items():
                if t1 != t2 and self.find_and_repair_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break
            else:
                teams[tk] = t1
                break
        # Handle case where odd number of teams and odd number of events required
        if len(teams) == 1 and events:
            tk = next(iter(teams.keys()))
            t = teams.pop(tk)
            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)
            for ek in event_keys:
                e1 = events[ek]
                if not schedule.conflicts(t, e1):
                    schedule.assign(t, e1)
                    events.pop(ek)
                    break
            else:
                teams[tk] = t
        return list(teams.values()), list(events.values())

    def find_and_repair_match(self, t1: int, t2: int, events: dict[int, int], s: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        event_keys = list(events.keys())
        self.rng.shuffle(event_keys)
        for ekey in event_keys:
            e1 = events[ekey]
            e2 = self.evt_prop.paired_idx[e1]
            if not (s.conflicts(t1, e1) or s.conflicts(t2, e2)):
                s.assign(t1, e1)
                s.assign(t2, e2)
                events.pop(ekey)
                return True
        return False
