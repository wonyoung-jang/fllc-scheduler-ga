"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from logging import getLogger
from math import comb

import numpy as np

from ..data_model.schedule import Population, Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: random.Random
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

        fronts = NSGA3._non_dominated_sort(population)
        last_idx = NSGA3._get_last_front_idx(fronts, population_size)

        selected = [p for i in range(last_idx) for p in fronts[i]]
        selected.extend(
            self._niching(
                fronts=fronts[: last_idx + 1],
                last_front=fronts[last_idx],
                k=population_size - len(selected),
            )
        )
        return {hash(p): p for p in selected}

    @staticmethod
    def _get_last_front_idx(fronts: list[Population], pop_size: int) -> int:
        """Determine which front is the last to be included."""
        total = 0
        for i, front in enumerate(fronts):
            total += len(front)
            if total >= pop_size:
                return i
        return len(fronts) - 1

    @staticmethod
    def _non_dominated_sort(pop: Population) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        _size = len(pop)
        _dom_list: list[list[int]] = [[] for _ in range(_size)]
        _dom_counts = [0] * _size
        _fits = [p.fitness for p in pop]

        for i, fi in enumerate(_fits):
            for j, fj in enumerate(_fits[i + 1 :], start=i + 1):
                if NSGA3._dominates(fi, fj):
                    _dom_list[i].append(j)
                    _dom_counts[j] += 1
                elif NSGA3._dominates(fj, fi):
                    _dom_list[j].append(i)
                    _dom_counts[i] += 1

        fronts: list[list[int]] = [[i for i in range(_size) if _dom_counts[i] == 0]]

        curr = 0
        while curr < len(fronts) and fronts[curr]:
            next_front = []
            for i in fronts[curr]:
                pop[i].rank = curr
                for j in _dom_list[i]:
                    _dom_counts[j] -= 1
                    if _dom_counts[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            curr += 1

        return [[pop[i] for i in fr] for fr in fronts]

    def _niching(self, fronts: list[Population], last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        all_schedules = [p for front in fronts for p in front]
        self._normalize_then_associate(all_schedules, last_front)
        counts = self._count(p.ref_point_idx for fr in fronts[:-1] for p in fr if p.ref_point_idx is not None)
        logger.debug("Counts of individuals per reference point: %s", counts)
        pool = dict(enumerate(last_front))
        selected = 0

        while selected < k and pool:
            found = False
            min_count = min(counts)
            d_counts = [i for i, c in enumerate(counts) if c == min_count]
            self.rng.shuffle(d_counts)
            for d in d_counts:
                clst_pool = {k: v for k, v in pool.items() if v.ref_point_idx == d}
                if not clst_pool:
                    continue
                cluster = [k for k, _ in sorted(clst_pool.items(), key=lambda p: p[1].distance_to_ref_point)]
                pi = cluster[0 if counts[d] == min_count else self.rng.randrange(0, len(cluster))]
                yield pool.pop(pi)
                found = True
                counts[d] += 1
                selected += 1
                if selected >= k:
                    break

            if not found and pool:
                pick = pool.pop(self.rng.choice(list(pool.keys())))
                if pick.ref_point_idx is not None:
                    counts[pick.ref_point_idx] += 1

                yield pick
                selected += 1

        logger.debug("Counts of individuals per reference point: %s", counts)
        logger.debug("Niching selected %d/%d individuals from the last front.", selected, k)

    def _normalize_then_associate(self, pop: Population, last_front: Population) -> None:
        """Normalize objectives then associate individuals with nearest reference points."""
        fits_all = np.fromiter((p.fitness for p in pop), dtype=self._dtype)
        fits_last = np.fromiter((p.fitness for p in last_front), dtype=self._dtype) if last_front else fits_all
        logger.debug("Fitness (all):\n%s", fits_all)
        logger.debug("Fitness (last):\n%s", fits_last)

        # Ensure 2D arrays: shape (N, m)
        if fits_all.ndim == 1:
            fits_all = fits_all.reshape(1, -1)
        if fits_last.ndim == 1:
            fits_last = fits_last.reshape(1, -1)

        # Ensure ref_points have matching objective dimension
        ref = self.ref_points
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)  # (R, m)

        if fits_all.shape[1] != ref.shape[1]:
            msg = (
                f"objective-dimension mismatch: fitness has {fits_all.shape[1]} objectives "
                f"but reference points have {ref.shape[1]}"
            )
            raise ValueError(msg)

        # Compute ideal (max) and nadir (min on the last front) and normalize
        ideal = fits_all.max(axis=0)  # shape (m,)
        nadir = fits_last.min(axis=0)  # shape (m,)
        span = ideal - nadir
        span[span == 0.0] = 1e-16  # avoid divide-by-zero
        logger.debug("\nIdeal: %s\nNadir: %s\nSpan: %s", ideal, nadir, span)

        for p, fit in zip(pop, fits_all, strict=True):
            p.normalized_fitness = (fit - nadir) / span
            dists = np.sum((self.ref_points - p.normalized_fitness) ** 2, axis=1)
            p.ref_point_idx = np.argmin(dists)
            p.distance_to_ref_point = dists[p.ref_point_idx]

    def _count(self, idx_to_count: Iterator[int]) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        counts = [0] * len(self.ref_points)
        for idx in idx_to_count:
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return counts

    @staticmethod
    def _dominates(fi: tuple[float, ...], fj: tuple[float, ...]) -> bool:
        """Check if schedule i dominates schedule j."""
        better_in_any = False
        for si, sj in zip(fi, fj, strict=True):
            if si < sj:
                return False
            if si > sj:
                better_in_any = True
        return better_in_any
