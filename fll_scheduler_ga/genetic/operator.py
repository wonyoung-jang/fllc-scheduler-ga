"""Genetic operators."""

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from logging import getLogger
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.constants import EPSILON, CrossoverOp, MutationOp, SelectionOp
from fll_scheduler_ga.domain.model import Schedule

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TournamentConfig

logger = getLogger(__name__)

########################################################################
###  NSGA-III
########################################################################


def calc_ref_points(n_obj: int, n_pop: int) -> np.ndarray:
    """Generate a set of structured reference points."""
    m = n_obj
    p = 1
    while comb(m + p - 1, m - 1) < n_pop:
        p += 1

    def _generate_coordinates() -> Iterator[np.ndarray]:
        for dividers in combinations(range(m + p - 1), m - 1):
            coords = np.zeros(m, dtype=float)
            prev = -1
            for i, divider in enumerate(dividers):
                coords[i] = divider - prev - 1
                prev = divider
            coords[-1] = m + p - 1 - dividers[-1] - 1
            yield (coords / p)

    coordinates = tuple(_generate_coordinates())
    points = np.array(coordinates, dtype=float)
    logger.debug("Generated %d reference points:\n%s", points.shape[0], points)
    return points


def calc_norm_sq_of_refs(points: np.ndarray) -> np.ndarray:
    """Calculate the squared norms of the reference points."""
    norm_sq = (points**2).sum(axis=1)
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
        # Pairwise comparisons using broadcasting
        all_ge = (fits[:, None, :] >= fits[None, :, :]).all(axis=2)
        any_gt = (fits[:, None, :] > fits[None, :, :]).any(axis=2)
        # dom[i,j] = True if i dominates j (>= on all and > on at least one)
        dom = np.logical_and(all_ge, any_gt)
        # Number of individuals that dominate j = sum over i dom[i,j]
        dom_count = dom.sum(axis=0)
        # Adjacency lists: who each i dominates
        assigned = np.zeros(n_fit, dtype=bool)
        fronts: list[np.ndarray] = []
        # Initial front: those not dominated by anybody
        current_front: np.ndarray = (dom_count == 0).nonzero()[0]
        assigned[current_front] = True
        fronts.append(current_front)
        n_ranked = current_front.size
        # Build subsequent fronts
        while n_ranked < n_pop and current_front.size > 0:
            # Sum of domination relationships from current_front to each j
            decrement = dom[current_front, :].sum(axis=0)
            dom_count = dom_count - decrement
            # Next front: those now not dominated by anybody
            next_front: np.ndarray = ((dom_count == 0) & (~assigned)).nonzero()[0]
            if next_front.size == 0:
                break
            assigned[next_front] = True
            fronts.append(next_front)
            n_ranked += next_front.size
            current_front = next_front
        return fronts

    def select(self, fits: np.ndarray, n_pop: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Select the next generation using NSGA-III principles."""
        fronts = self.get_fronts(fits, n_pop)
        last_idx = len(fronts) - 1
        selected_indices = np.array([i for f in fronts for i in f], dtype=int)
        selected_fits = fits[selected_indices]
        refs, distances = self.norm_and_associate(selected_fits)
        if len(fronts) == 1:
            fronts[0] = self.rng.permutation(selected_indices)[:n_pop]
            fronts = tuple(fronts)
            flat = np.concatenate(fronts)
            ranks = self.ranks_from_fronts(fronts, fits.shape[0])
            return fronts, flat, ranks[flat]
        last_front_indices = fronts[last_idx]
        n_last_front = last_front_indices.size
        fronts = fronts[:last_idx]
        selected = selected_indices[:-n_last_front]
        n_remaining = n_pop - selected.size
        niche_selected = refs[:-n_last_front]
        counts = self.count(niche_selected)
        niches = self.niche(
            counts=counts,
            n_last_front=n_last_front,
            n_remaining=n_remaining,
            niche_refs=refs[-n_last_front:],
            niche_dists=distances[-n_last_front:],
        )
        last_front_indices = last_front_indices[niches]
        fronts.append(last_front_indices)
        fronts = tuple(fronts)
        flat = np.concatenate(fronts)
        ranks = self.ranks_from_fronts(fronts, fits.shape[0])
        return fronts, flat, ranks[flat]

    def ranks_from_fronts(self, fronts: tuple[np.ndarray, ...], n_individuals: int) -> np.ndarray:
        """Assign ranks to individuals based on their fronts."""
        ranks = np.full(n_individuals, fill_value=-1, dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
        return ranks

    def niche(
        self, counts: np.ndarray, n_last_front: int, n_remaining: int, niche_refs: np.ndarray, niche_dists: np.ndarray
    ) -> np.ndarray:
        """Select k individuals from the last front using a niching mechanism."""
        # Mask of individuals in the last front still available for selection
        mask = np.full(n_last_front, fill_value=True, dtype=bool)
        n_selected = 0
        while n_selected < n_remaining:
            # All reference points associated with individuals still available
            available_refs = np.unique(niche_refs[mask])
            ref_counts = counts[available_refs]
            # Minimum count among those reference points
            min_count = ref_counts.min()
            # Number of individuals to select from this niche
            n_select = n_remaining - n_selected
            niche_indices = available_refs[(ref_counts == min_count).nonzero()[0]]
            niche_indices = niche_indices[self.rng.permutation(niche_indices.size)[:n_select]]
            for niche_idx in niche_indices:
                # Indices of individuals in this niche still available
                next_i = ((niche_refs == niche_idx) & mask).nonzero()[0]
                self.rng.shuffle(next_i)
                index = next_i[niche_dists[next_i].argmin()] if counts[niche_idx] == 0 else next_i[0]
                mask[index] = False
                counts[niche_idx] += 1
                n_selected += 1
                if n_selected >= n_remaining:
                    break
        # Return the masked indices
        return (~mask).nonzero()[0]

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
        residuals = norm[:, None, :] - proj
        dists: np.ndarray = np.linalg.norm(residuals, axis=2)
        min_dists = dists.min(axis=1)
        # Mask tied positions with random values
        ties = dists == min_dists[:, None]
        rand_matrix = self.rng.random(dists.shape)
        rand_matrix[~ties] = -1.0
        chosen_refs = rand_matrix.argmax(axis=1)  # Index of chosen ref per individual
        return chosen_refs, min_dists

    def count(self, niche_selected: np.ndarray) -> np.ndarray:
        """Count how many individuals are associated with each reference point."""
        counts = np.zeros(self.n_refs, dtype=int)
        indices, count = np.unique(niche_selected, return_counts=True)
        counts[indices] = count
        return counts


########################################################################
###  Selection
########################################################################


@dataclass(slots=True)
class Selection(ABC):
    """Abstract base class for selection operators in genetic algorithms."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @abstractmethod
    def select(self, n: int, k: int) -> np.ndarray: ...


class RandomSelect(Selection):
    """Random selection of individuals from the population."""

    def __str__(self) -> str:
        """Return a string representation of the selection operator."""
        return SelectionOp.RANDOM_SELECT

    def select(self, n: int, k: int = 2) -> np.ndarray:
        """Select individuals from the population to form the next generation.

        Args:
            n (int): The population size to select from.
            k (int): The number to select.

        Returns:
            np.ndarray: The indices of the selected individuals.

        """
        if k == 2:
            # Two random indices
            i1 = self.rng.integers(0, n)
            i2 = self.rng.integers(0, n)
            # Ensure distinct
            while i1 == i2:
                i2 = self.rng.integers(0, n)
            return np.array((i1, i2), dtype=int)
        choices = np.arange(n)
        self.rng.shuffle(choices)
        return choices[:k]


########################################################################
###  Crossover
########################################################################
def build_crossovers(
    rng: np.random.Generator,
    crossover_types: tuple[str, ...],
    crossover_ks: tuple[int, ...],
    evt_repo: EventRepository,
    evt_prop: EventProperties,
) -> tuple[Crossover, ...]:
    """Build and return a tuple of crossover operators based on the configuration."""
    if not crossover_types:
        logger.warning("No crossover types enabled in the configuration. Crossover will not occur.")
        return ()
    crossover_factory: dict[str, Callable] = {
        CrossoverOp.K_POINT: lambda p, k: KPoint(**p, k=k),
        CrossoverOp.SCATTERED: Scattered,
        CrossoverOp.UNIFORM: Uniform,
        CrossoverOp.ROUND_TYPE_CROSSOVER: RoundTypeCrossover,
        CrossoverOp.TIMESLOT_CROSSOVER: TimeSlotCrossover,
        CrossoverOp.LOCATION_CROSSOVER: LocationCrossover,
    }
    params = {"evt_repo": evt_repo, "evt_prop": evt_prop, "rng": rng}

    def _generate_crossovers() -> Iterator[Crossover]:
        for crossover_name in crossover_types:
            if crossover_name not in crossover_factory:
                msg = f"Unknown crossover type in config: {crossover_name}"
                raise ValueError(msg)
            if crossover_name == CrossoverOp.K_POINT:
                if crossover_ks:
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

    evt_repo: EventRepository
    evt_prop: EventProperties
    rng: np.random.Generator
    _evts: np.ndarray = field(init=False)
    _n_evts: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the crossover operator."""
        self._evts = self.evt_repo.singles_or_side1_idx
        self._n_evts = self._evts.shape[0]

    @abstractmethod
    def cross(self, parents: Iterator[Schedule]) -> Iterator[Schedule]: ...

    def _create_child(self, p1: np.ndarray, p2: np.ndarray, p1_genes: np.ndarray, p2_genes: np.ndarray) -> Schedule:
        """Create a child schedule from two parents."""
        child = Schedule(origin=f"(C | {self!s})")
        self.assign_from_p1(child, p1, p1_genes)
        self.assign_from_p2(child, p2, p2_genes)
        return child

    def assign_from_p1(self, child: Schedule, p1: np.ndarray, p1_genes: np.ndarray) -> None:
        """Assign genes."""
        p1_gene_pairs = self.evt_prop.paired_idx[p1_genes]
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
        p2_genes_pairs = self.evt_prop.paired_idx[p2_genes]
        p2_genes_rt = self.evt_prop.roundtype_idx[p2_genes]
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


class EventCrossover(Crossover):
    """Abstract base class for crossover operators in the FLL Scheduler GA."""

    def __str__(self) -> str:
        """Return a string representation of the crossover operator."""
        if (k := getattr(self, "k", None)) is not None:
            return f"{self.__class__.__name__}(k={k})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def get_genes(self) -> Iterable[np.ndarray]: ...

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
        if not 1 <= self.k < self._n_evts:
            logger.warning("Invalid k value for KPoint crossover: %d. Setting k to 1.", self.k)
            self.k = 1

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for KPoint crossover."""
        # Single-point crossover
        if self.k == 1:
            split = self.rng.integers(1, self._n_evts)
            return self._evts[:split], self._evts[split:]
        # Multi-point crossover
        splits = self.rng.choice(self._n_evts - 1, size=self.k, replace=False) + 1
        mask = np.zeros(self._n_evts, dtype=bool)
        mask[splits] = True
        np.bitwise_xor.accumulate(mask, out=mask)
        return self._evts[mask], self._evts[~mask]


class Scattered(EventCrossover):
    """Scattered crossover operator for genetic algorithms.

    Shuffled indices split parent 50/50.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Scattered crossover."""
        permuted_indices = self.rng.permutation(self._evts)
        return np.array_split(permuted_indices, 2)


class Uniform(EventCrossover):
    """Uniform crossover operator for genetic algorithms.

    Each gene is chosen from either parent by flipping a coin for each gene.
    The main difference with Scattered, is Scattered guarantees close to 50/50 splits.
    Uniform may result in more imbalanced splits.
    """

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Uniform crossover."""
        mask = self.rng.random(self._n_evts) < 0.5
        return self._evts[mask], self._evts[~mask]


@dataclass(slots=True)
class StructureCrossover(EventCrossover):
    """Structure-based crossover operator for genetic algorithms.

    Each gene is chosen based on a specific structure of the event.
    """

    _lookup: np.ndarray = field(init=False)
    _structure: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        super().__post_init__()
        eventmap = defaultdict(list)
        for key, e in zip(self._get_group_keys(), self._evts, strict=True):
            eventmap[key].append(e)
        unique_ids = np.array(sorted(eventmap.keys()))
        n_ids = unique_ids.shape[0]
        max_len = max(len(evts) for evts in eventmap.values())
        self._lookup = np.full((n_ids, max_len), -1, dtype=int)
        for i, uid in enumerate(unique_ids):
            evts = eventmap[uid]
            self._lookup[i, : len(evts)] = evts
        self._structure = np.arange(n_ids)

    def get_genes(self) -> Iterable[np.ndarray]:
        """Get the genes for Structure-based crossover."""
        self.rng.shuffle(self._structure)
        p1, p2 = np.array_split(self._structure, indices_or_sections=2, axis=0)
        p1_indices = self._lookup[p1]
        p2_indices = self._lookup[p2]
        return p1_indices[p1_indices >= 0], p2_indices[p2_indices >= 0]

    @abstractmethod
    def _get_group_keys(self) -> np.ndarray: ...


class RoundTypeCrossover(StructureCrossover):
    """TournamentRound type crossover operator for genetic algorithms.

    Each gene is chosen based on the round type of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.evt_prop.roundtype_idx[self._evts]


class TimeSlotCrossover(StructureCrossover):
    """Time slot crossover operator for genetic algorithms.

    Each gene is chosen based on the time slot of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.evt_prop.timeslot_idx[self._evts]


class LocationCrossover(StructureCrossover):
    """Location crossover operator for genetic algorithms.

    Each gene is chosen based on the location of the event.
    """

    def _get_group_keys(self) -> np.ndarray:
        """Get all group keys for the events."""
        return self.evt_prop.loc_idx[self._evts]


########################################################################
###  Mutation
########################################################################
type Match = tuple[int, int, int, int]


def build_mutations(
    rng: np.random.Generator, mutation_types: tuple[str, ...], evt_repo: EventRepository, evt_prop: EventProperties
) -> tuple[Mutation, ...]:
    """Build and return a tuple of mutation operators based on the configuration."""
    if not mutation_types:
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return ()
    mutation_factory: dict[str, Callable[[dict], Mutation]] = {
        # SwapMatchMutation variants
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda p: SwapMatchMutation(
            **p, same_timeslot=False, same_location=False
        ),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda p: SwapMatchMutation(**p, same_timeslot=False, same_location=True),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda p: SwapMatchMutation(**p, same_timeslot=True, same_location=False),
        # SwapTeamMutation variants
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda p: SwapTeamMutation(
            **p, same_timeslot=False, same_location=False
        ),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda p: SwapTeamMutation(**p, same_timeslot=False, same_location=True),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda p: SwapTeamMutation(**p, same_timeslot=True, same_location=False),
        # SwapTableSideMutation variant
        MutationOp.SWAP_TABLE_SIDE: lambda p: SwapTableSideMutation(**p, same_timeslot=True, same_location=True),
        # TimeSlotSequenceMutation variants
        MutationOp.INVERSION: lambda p: InversionMutation(**p),
        MutationOp.SCRAMBLE: lambda p: ScrambleMutation(**p),
    }
    params = {"rng": rng, "evt_repo": evt_repo, "evt_prop": evt_prop}

    def _generate_mutations() -> Iterator[Mutation]:
        for mutation_name in mutation_types:
            if mutation_name not in mutation_factory:
                msg = f"Unknown mutation type in config: '{mutation_name}'"
                raise ValueError(msg)
            yield mutation_factory[mutation_name](params)

    return tuple(_generate_mutations())


@dataclass(slots=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: np.random.Generator
    evt_repo: EventRepository
    evt_prop: EventProperties

    @abstractmethod
    def mutate(self, schedule: Schedule) -> bool: ...


@dataclass(slots=True)
class SwapMutation(Mutation):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    same_timeslot: bool
    same_location: bool
    swap_candidates: list[tuple[tuple[int, ...], ...]] = field(default_factory=list)
    n_swap_candidates: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.swap_candidates.extend(self.init_swap_candidates())
        self.n_swap_candidates = len(self.swap_candidates)
        logger.debug("Initialized %d swap candidates for %s", self.n_swap_candidates, str(self))

    def init_swap_candidates(self) -> Iterator[tuple[tuple[int, ...], ...]]:
        """Precompute any necessary data before mutation."""
        _is_same_ts_and_loc = self.same_timeslot and self.same_location
        for match_list in self.evt_repo.matches.values():
            for match1, match2 in itertools.combinations(match_list, 2):
                e1a, _ = match1
                e2a, _ = match2
                _ts_cond = (self.evt_prop.timeslot_idx[e1a] == self.evt_prop.timeslot_idx[e2a]) == self.same_timeslot
                _loc_cond = (self.evt_prop.loc_idx[e1a] == self.evt_prop.loc_idx[e2a]) == self.same_location
                _is_swap_valid = _ts_cond and _loc_cond
                if _is_same_ts_and_loc or _is_swap_valid:
                    yield (match1, match2)

    @abstractmethod
    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]: ...


class SwapTeamMutation(SwapMutation):
    """Mutation operator for swapping single team between two matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_TEAM_SAME_LOCATION,
            (False, True): MutationOp.SWAP_TEAM_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

    def mutate(self, schedule: Schedule) -> bool:
        """Swap one team from two different matches."""
        if self.n_swap_candidates <= 0:
            return False
        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False
        e1a, _, t1a, _ = match1_data
        e2a, _, t2a, _ = match2_data
        schedule.swap_assignment(t1a, e1a, e2a)
        schedule.swap_assignment(t2a, e2a, e1a)
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        shuffled_idx = self.rng.permutation(self.n_swap_candidates)
        for idx in shuffled_idx:
            idx: int
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data
            t1a, t1b = schedule.schedule[e1a], schedule.schedule[e1b]
            t2a, t2b = schedule.schedule[e2a], schedule.schedule[e2b]
            match_team_ids = {t1a, t1b, t2a, t2b}
            if (
                len(match_team_ids) < 4
                or schedule.conflicts(t1a, e2a, ignore=e1a)
                or schedule.conflicts(t2a, e1a, ignore=e2a)
            ):
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

    def mutate(self, schedule: Schedule) -> bool:
        """Swap two entire matches."""
        if self.n_swap_candidates <= 0:
            return False
        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False
        e1a, e1b, t1a, t1b = match1_data
        e2a, e2b, t2a, t2b = match2_data
        none_in_m1 = -1 in (t1a, t1b)
        none_in_m2 = -1 in (t2a, t2b)
        if not none_in_m1:
            schedule.swap_assignment(t1a, e1a, e2a)
            schedule.swap_assignment(t1b, e1b, e2b)
        if not none_in_m2:
            schedule.swap_assignment(t2a, e2a, e1a)
            schedule.swap_assignment(t2b, e2b, e1b)
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        shuffled_idx = self.rng.permutation(self.n_swap_candidates)
        for idx in shuffled_idx:
            idx: int
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data
            t1a, t1b = schedule.schedule[e1a], schedule.schedule[e1b]
            if -1 not in (t1a, t1b) and (
                schedule.conflicts(t1a, e2a, ignore=e1a) or schedule.conflicts(t1b, e2b, ignore=e1b)
            ):
                continue
            t2a, t2b = schedule.schedule[e2a], schedule.schedule[e2b]
            if -1 not in (t2a, t2b) and (
                schedule.conflicts(t2a, e1a, ignore=e2a) or schedule.conflicts(t2b, e1b, ignore=e2b)
            ):
                continue
            return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)
        return None, None


class SwapTableSideMutation(SwapMutation):
    """Mutation operator for swapping the sides of two tables in a match."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SWAP_TABLE_SIDE

    def mutate(self, schedule: Schedule) -> bool:
        """Swap the sides of two tables in a match."""
        if self.n_swap_candidates <= 0:
            return False
        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False
        e1a, e1b, t1a, t1b = match1_data
        schedule.swap_assignment(t1a, e1a, e1b)
        schedule.swap_assignment(t1b, e1b, e1a)
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get one match to swap sides in the schedule schedule."""
        idx = self.rng.integers(0, self.n_swap_candidates)
        match1_data, match2_data = self.swap_candidates[idx]
        e1a, e1b = match1_data
        e2a, e2b = match2_data
        t1a, t1b = schedule.schedule[e1a], schedule.schedule[e1b]
        t2a, t2b = schedule.schedule[e2a], schedule.schedule[e2b]
        return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)


@dataclass(slots=True)
class TimeSlotSequenceMutation(Mutation):
    """Abstract base class for mutations that permute assignments within a single timeslot."""

    timeslot_candidates: dict[tuple[int, int], list[tuple[int, ...]]] = field(init=False)
    timeslot_keys: tuple[tuple[int, int], ...] = field(init=False)
    key_to_tpr: dict[tuple[int, int], int] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.timeslot_candidates, self.key_to_tpr = self.init_candidates()
        self.timeslot_keys = tuple(self.timeslot_candidates.keys())

    @abstractmethod
    def permute_singles(self, items: list[int]) -> Iterator[int]: ...
    @abstractmethod
    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]: ...

    def init_candidates(self) -> tuple[dict[tuple[int, int], list[tuple[int, ...]]], dict[tuple[int, int], int]]:
        """Precompute candidate events for each timeslot."""
        timeslot_data: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        keys_to_tpr: dict[tuple[int, int], int] = {}
        for key, events in self.evt_repo.timeslots.items():
            candidates = [e for e in events if self.evt_prop.loc_side[e] == 1 or self.evt_prop.paired_idx[e] == -1]
            timeslot_data[key] = [(e, self.evt_prop.paired_idx[e]) for e in candidates]
            keys_to_tpr[key] = self.evt_prop.teams_per_round[events[0]]
        return timeslot_data, keys_to_tpr

    def get_candidates(self) -> tuple[list[tuple[int, ...]], int]:
        """Get a list of candidate events for mutation within a specific timeslot."""
        indices = np.arange(len(self.timeslot_keys))
        self.rng.shuffle(indices)
        idx = indices[0]
        key = self.timeslot_keys[idx]
        candidates = self.timeslot_candidates[key]
        tpr = self.key_to_tpr[key]
        return candidates, tpr

    def mutate(self, schedule: Schedule) -> bool:
        """Find a suitable timeslot and round type, then permute assignments."""
        candidates, tpr = self.get_candidates()
        if tpr == 1:
            return self.mutate_singles(schedule, candidates)
        if tpr == 2:
            return self.mutate_matches(schedule, candidates)
        return False

    def mutate_singles(self, schedule: Schedule, candidates: list[tuple[int, ...]]) -> bool:
        """Permute team assignments for single-team events."""
        old_ids = [schedule.schedule[e] for e, _ in candidates]
        new_ids = self.permute_singles(old_ids)
        for (event, _), old_team, new_team in zip(candidates, old_ids, new_ids, strict=True):
            if old_team != new_team:
                schedule.unassign(old_team, event)
                schedule.assign(new_team, event)
        return True

    def mutate_matches(self, schedule: Schedule, candidates: list[tuple[int, ...]]) -> bool:
        """Permute team assignments for match-based events."""
        matches: list[tuple[int, ...]] = []
        old_ids: list[tuple[int, ...]] = []
        for e1, e2 in candidates:
            t1, t2 = schedule.schedule[e1], schedule.schedule[e2]
            matches.append((e1, e2))
            old_ids.append((t1, t2))
        new_ids = self.permute_matches(old_ids)
        for (e1, e2), old_id_pair, new_id_pair in zip(matches, old_ids, new_ids, strict=True):
            if old_id_pair != new_id_pair:
                old_t1, old_t2 = old_id_pair
                schedule.unassign(old_t1, e1)
                schedule.unassign(old_t2, e2)
                new_t1, new_t2 = new_id_pair
                schedule.assign(new_t1, e1)
                schedule.assign(new_t2, e2)
        return True


class InversionMutation(TimeSlotSequenceMutation):
    """Inverts a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.INVERSION

    def permute_singles(self, items: list[int]) -> Iterator[int]:
        """Invert a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return reversed(items[:])

    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Invert a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return reversed([tuple(reversed(pair)) for pair in items])


class ScrambleMutation(TimeSlotSequenceMutation):
    """Scrambles a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SCRAMBLE

    def permute_singles(self, items: list[int]) -> Iterator[int]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return iter(self.rng.permutation(items))

    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return (tuple(self.rng.permutation(pair)) for pair in items)


########################################################################
###  Repair
########################################################################


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    config: TournamentConfig
    event_properties: EventProperties
    rng: np.random.Generator
    _repair_map: dict[int, Any] = field(init=False)
    _rt_to_tpr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._repair_map = {1: self.repair_singles, 2: self.repair_matches}
        max_rt = max(self.config.round_idx_to_tpr.keys())
        self._rt_to_tpr = np.zeros(max_rt + 1, dtype=int)
        for rt, tpr in self.config.round_idx_to_tpr.items():
            self._rt_to_tpr[rt] = tpr

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.
        """
        if schedule.get_size() == self.config.total_slots_required:
            return True
        teams, events = self.get_rt_tpr_maps(schedule)
        return self.iterative_repair(schedule, teams, events)

    def iterative_repair(
        self, schedule: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        while schedule.get_size() < self.config.total_slots_required:
            if self._attempt_repair_step(teams, events, schedule):
                return True
            self._unassign_and_requeue_event(teams, events, schedule)
        return schedule.get_size() == self.config.total_slots_required

    def _attempt_repair_step(
        self, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]], schedule: Schedule
    ) -> bool:
        """Attempt to apply a repair function for the current round type.

        Returns True if the schedule is considered resolved for this step.
        """
        for key, teams_for_rt in teams.items():
            _, tpr = key
            if not (events_for_rt := events.get(key)):
                return True
            if not (repair_fn := self._repair_map.get(tpr)):
                msg = f"No assignment function for teams per round: {tpr}"
                raise ValueError(msg)
            _teams, _events = repair_fn(
                teams=dict(enumerate(teams_for_rt)), events=dict(enumerate(events_for_rt)), schedule=schedule
            )
            teams[key] = _teams
            events[key] = _events
            if _teams:
                return False
        return True

    def _unassign_and_requeue_event(
        self, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]], schedule: Schedule
    ) -> None:
        """Select a random scheduled event, handle pairing logic, and move it back to the queue."""
        event_indices = schedule.scheduled_events()
        self.rng.shuffle(event_indices)
        primary_event = event_indices[0]
        e_rt_idx = self.event_properties.roundtype_idx[primary_event]
        ek = (e_rt_idx, self.config.round_idx_to_tpr[e_rt_idx])
        paired_event = self.event_properties.paired_idx[primary_event]
        loc_side = self.event_properties.loc_side[primary_event]
        if paired_event != -1:
            e1, e2 = (paired_event, primary_event) if loc_side == 2 else (primary_event, paired_event)
        else:
            e1, e2 = primary_event, None
        t1 = schedule.schedule[e1]
        events[ek].append(e1)
        teams[ek].append(t1)
        schedule.unassign(t1, e1)
        if e2 is not None and (t2 := schedule.schedule[e2]) != -1:
            teams[ek].append(t2)
            schedule.unassign(t2, e2)

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
            paired = self.event_properties.paired_idx[unscheduled]
            sides = self.event_properties.loc_side[unscheduled]
            # Mask for valid repair candidates (singles or side 1 of matches)
            mask = (paired == -1) | (sides == 1)
            valid_events = unscheduled[mask]
            if valid_events.size > 0:
                valid_rts = self.event_properties.roundtype_idx[valid_events]
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
        while len(teams) >= 1:
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tkey = team_keys[0]
            t = teams.pop(tkey)
            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)
            for ekey in event_keys:
                e = events[ekey]
                if schedule.conflicts(t, e):
                    continue
                schedule.assign(t, e)
                events.pop(ekey)
                break
            else:
                teams[tkey] = t
                break
        return list(teams.values()), list(events.values())

    def repair_matches(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign match events to teams that need them."""
        while len(teams) >= 2:
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tkey = team_keys[0]
            t1 = teams.pop(tkey)
            for i, t2 in teams.items():
                if t1 != t2 and self.find_and_repair_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break
            else:
                teams[tkey] = t1
                break
        # Handle case where odd number of teams and odd number of events required
        if len(teams) == 1 and events:
            tkey = next(iter(teams.keys()))
            t_solo = teams.pop(tkey)
            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)
            for ekey in event_keys:
                e1 = events[ekey]
                if not schedule.conflicts(t_solo, e1):
                    schedule.assign(t_solo, e1)
                    events.pop(ekey)
                    break
            else:
                teams[tkey] = t_solo
        return list(teams.values()), list(events.values())

    def find_and_repair_match(self, t1: int, t2: int, events: dict[int, int], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        event_keys = list(events.keys())
        self.rng.shuffle(event_keys)
        for ekey in event_keys:
            e1 = events[ekey]
            e2 = self.event_properties.paired_idx[e1]
            if not (schedule.conflicts(t1, e1) or schedule.conflicts(t2, e2)):
                schedule.assign(t1, e1)
                schedule.assign(t2, e2)
                events.pop(ekey)
                return True
        return False
