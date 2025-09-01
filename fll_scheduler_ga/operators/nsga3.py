"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from logging import getLogger
from math import comb
from random import Random

import numpy as np

from ..data_model.schedule import Population, Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: Random
    num_objectives: int
    population_size: int
    ref_points: np.ndarray = field(init=False, repr=False)
    _dtype: np.dtype = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to generate reference points."""
        self._initialize_reference_points()

    def _gen_ref_points(self) -> Iterator[list[float]]:
        """Generate a set of structured reference points."""
        m = self.num_objectives
        p = 1
        while comb(p + m - 1, m - 1) < self.population_size:
            p += 1

        for c in combinations(range(p + m - 1), m - 1):
            coords = []
            prev = -1
            for idx in c:
                coords.append(idx - prev - 1)
                prev = idx
            coords.append(p + m - 1 - c[-1] - 1)
            yield [x / p for x in coords]

    def _initialize_reference_points(self) -> None:
        """Generate a set of structured reference points."""
        self._dtype = np.dtype((float, self.num_objectives))
        self.ref_points = np.fromiter(self._gen_ref_points(), dtype=self._dtype)
        logger.debug("Generated %d reference points:\n%s", len(self.ref_points), self.ref_points)

    def select(self, population: Population | None, population_size: int) -> dict[int, Schedule]:
        """Select the next generation using NSGA-III principles."""
        if not isinstance(population, list):
            population = list(population)

        fronts = NSGA3._non_dominated_sort(population, population_size)
        if len(fronts) - 1 <= 0:
            return {hash(p): p for p in fronts[0][:population_size]}

        selected = [p for fr in fronts[:-1] for p in fr]
        selected.extend(
            self._niching(
                fronts=fronts,
                last_front=fronts[-1],
                k=population_size - len(selected),
            )
        )
        return {hash(p): p for p in selected}

    @staticmethod
    def _non_dominated_sort(pop: Population, pop_size: int) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        _size = len(pop)
        _dom_list: list[list[int]] = [[] for _ in range(_size)]
        _dom_counts = [0] * _size
        _fits = [p.fitness for p in pop]

        for i, fi in enumerate(_fits):
            for j, fj in enumerate(_fits[i + 1 :], start=i + 1):
                if dominates(fi, fj):
                    _dom_list[i].append(j)
                    _dom_counts[j] += 1
                elif dominates(fj, fi):
                    _dom_list[j].append(i)
                    _dom_counts[i] += 1

        fronts: list[list[int]] = [[]]
        pop_count = 0
        first_front = (i for i in range(_size) if _dom_counts[i] == 0)
        for i in first_front:
            pop_count += 1
            pop[i].rank = 0
            fronts[0].append(i)

        curr = 0
        while curr < len(fronts) and fronts[curr] and pop_count < pop_size:
            next_front = []
            for i in fronts[curr]:
                for j in _dom_list[i]:
                    _dom_counts[j] -= 1
                    if _dom_counts[j] == 0:
                        pop[j].rank = curr + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                pop_count += len(next_front)
            curr += 1

        return [[pop[i] for i in fr] for fr in fronts]

    def _niching(self, fronts: list[Population], last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        self._normalize_then_associate(fronts, last_front)
        niche_counts = self._count(p.ref_point for fr in fronts[:-1] for p in fr)
        pool = dict(enumerate(last_front))
        selected = 0

        while selected < k:
            next_pool_counts = {i.ref_point for i in pool.values()}
            next_niche_counts = [(i, c) for i, c in enumerate(niche_counts) if i in next_pool_counts]
            min_count = min((c for _, c in next_niche_counts), default=None)
            next_niches = [i for i, c in next_niche_counts if c == min_count]
            if not next_niches:
                break

            self.rng.shuffle(next_niches)
            for next_niche in next_niches:
                sorted_pool = sorted(pool.items(), key=lambda p: p[1].ref_point_distance)
                cluster = [i for i, v in sorted_pool if v.ref_point == next_niche]
                if not cluster:
                    continue

                yield pool.pop(cluster[0])
                niche_counts[next_niche] += 1
                selected += 1
                if selected >= k:
                    break

    def _normalize_then_associate(self, fronts: list[Population], last_front: Population) -> None:
        """Normalize objectives then associate individuals with nearest reference points."""
        pop = [p for fr in fronts for p in fr]
        fits_all = np.fromiter((p.fitness for p in pop), dtype=self._dtype)
        fits_last = np.fromiter((p.fitness for p in last_front), dtype=self._dtype)

        # Compute ideal (max) and nadir (min on the last front) and normalize
        ideal = fits_all.max(axis=0)  # shape (m,)
        nadir = fits_last.min(axis=0)  # shape (m,)
        span = ideal - nadir
        span[span == 0.0] = 1e-12  # avoid divide-by-zero
        norm_fits: np.ndarray = (fits_all - nadir) / span
        u = np.tile(self.ref_points, (len(norm_fits), 1))
        v = np.repeat(norm_fits, len(self.ref_points), axis=0)
        norm_u = np.linalg.norm(u, axis=1)
        scalar_proj = np.sum(u * v, axis=1) / norm_u
        proj = scalar_proj[:, None] * u / norm_u[:, None]
        val = np.linalg.norm(proj - v, axis=1)
        dists = np.reshape(val, (len(norm_fits), len(self.ref_points)))

        for p, d in zip(pop, dists, strict=True):
            p.ref_point = int(np.argmin(d))
            p.ref_point_distance = float(d[np.arange(d.shape[0]) == p.ref_point])

    def _count(self, idx_to_count: Iterator[int]) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        counts = [0] * len(self.ref_points)
        for idx in idx_to_count:
            counts[idx] += 1
        return counts


@cache
def dominates(fi: tuple[float, ...], fj: tuple[float, ...]) -> bool:
    """Check if schedule i dominates schedule j."""
    better_in_any = False
    for si, sj in zip(fi, fj, strict=True):
        if si < sj:
            return False
        if si > sj:
            better_in_any = True
    return better_in_any
