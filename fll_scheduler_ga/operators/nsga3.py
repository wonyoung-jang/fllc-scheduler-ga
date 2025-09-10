"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from logging import getLogger
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from random import Random

    from ..data_model.schedule import Population, Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: Random
    num_objectives: int
    total_size: int
    island_size: int
    ref_points: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to generate reference points."""
        self.init_ref_points()

    def init_ref_points(self) -> None:
        """Generate a set of structured reference points."""
        m = self.num_objectives
        p = 1
        while comb(p + m - 1, m - 1) < self.total_size:
            p += 1

        pts = []
        for c in combinations(range(p + m - 1), m - 1):
            coords = []
            prev = -1
            for idx in c:
                coords.append(idx - prev - 1)
                prev = idx
            coords.append(p + m - 1 - c[-1] - 1)
            pts.append([x / p for x in coords])

        if not pts:
            pts = [[1.0 / m] * m]

        self.ref_points = np.asarray(pts, dtype=float)
        logger.debug("Generated %d reference points:\n%s", len(self.ref_points), self.ref_points)

    def get_last_front_idx(self, fronts: list[Population], size: int) -> int:
        """Determine which front is the last to be included."""
        total = 0
        for i, front in enumerate(fronts):
            total += len(front)
            if total >= size:
                return i
        return len(fronts) - 1

    def select(self, population: Population | Any, size: int | None = None) -> dict[int, Schedule]:
        """Select the next generation using NSGA-III principles."""
        size = size or self.island_size
        if not isinstance(population, list):
            population = list(population)

        fronts = self.non_dominated_sort(population)
        last_idx = self.get_last_front_idx(fronts, size)
        pop = [p for f in fronts[: last_idx + 1] for p in f]
        self.norm_and_associate(pop)
        if len(fronts) == 1:
            selected = sorted(pop, key=lambda p: p.ref_distance)
            return {hash(p): p for p in selected[:size]}

        last_front = fronts[last_idx]
        selected = [p for f in fronts[:last_idx] for p in f]
        counts = self.count(p.ref_point for p in selected)
        selected.extend(
            self.niche(
                counts=counts,
                last_front=last_front,
                k=size - len(selected),
            )
        )
        return {hash(p): p for p in selected}

    def non_dominated_sort(self, pop: Population) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        n = len(pop)
        dom_list: list[list[int]] = [[] for _ in range(n)]
        dom_count = np.zeros(n, dtype=int)
        fits = [p.fitness for p in pop]

        for i, fi in enumerate(fits):
            for j, fj in enumerate(fits[i + 1 :], start=i + 1):
                if dominates(fi, fj):
                    dom_list[i].append(j)
                    dom_count[j] += 1
                elif dominates(fj, fi):
                    dom_list[j].append(i)
                    dom_count[i] += 1

        fronts: list[list[int]] = [[]]
        for i in range(n):
            if dom_count[i] == 0:
                fronts[0].append(i)
                pop[i].rank = 0

        rank = 0
        while rank < len(fronts) and fronts[rank]:
            next_rank = rank + 1
            next_front: list[int] = []
            for i in fronts[rank]:
                for j in dom_list[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        pop[j].rank = next_rank
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            rank = next_rank

        return [[pop[i] for i in fr] for fr in fronts]

    def niche(self, counts: np.ndarray, last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        sample = self.rng.sample
        choice = self.rng.choice
        pool: dict[int, Schedule] = dict(enumerate(last_front))
        selected = 0

        while selected < k and pool:
            available_refs = {s.ref_point for s in pool.values()}
            available_counts = [(i, counts[i]) for i in available_refs]
            min_count = min(c for _, c in available_counts)
            min_refs = [i for i, c in available_counts if c == min_count]
            if not min_refs:
                break

            for niche in sample(min_refs, len(min_refs)):
                if not (
                    cluster := sorted(
                        ((i, s) for i, s in pool.items() if s.ref_point == niche),
                        key=lambda k: k[1].ref_distance,
                    )
                ):
                    continue

                if counts[niche] == min_count:
                    yield pool.pop(cluster[0][0])
                else:
                    yield pool.pop(choice(cluster)[0])

                counts[niche] += 1
                selected += 1
                if selected >= k:
                    break

        if selected < k and pool:
            remaining = k - selected
            keys = list(pool.keys())
            for key in sample(keys, min(remaining, len(keys))):
                yield pool.pop(key)

    def norm_and_associate(self, pop: Population) -> None:
        """Normalize objectives then associate individuals with nearest reference points."""
        fits = np.asarray([p.fitness for p in pop], dtype=float)
        epsilon = 1e-12
        ideal = fits.max(axis=0)
        nadir = fits.min(axis=0)
        span = ideal - nadir
        span[span == 0] = epsilon
        norm = (fits - nadir) / span

        pts = self.ref_points
        norm_sq = np.sum(pts**2, axis=1)  # (H,)
        norm_sq[norm_sq == 0] = epsilon

        coeffs = (norm @ pts.T) / norm_sq
        proj = coeffs[:, :, None] * pts[None, :, :]
        residuals = norm[:, None, :] - proj

        dists = np.linalg.norm(residuals, axis=2)
        min_dists = dists.min(axis=1)
        ties = dists == min_dists[:, None]
        choice = self.rng.choice
        for i, s in enumerate(pop):
            candidates = np.nonzero(ties[i])[0]
            s.ref_point = choice(candidates)
            s.ref_distance = min_dists[i]

    def count(self, idx_to_count: Iterator[int]) -> np.ndarray:
        """Count how many individuals are associated with each reference point."""
        counts = np.zeros(len(self.ref_points), dtype=int)
        total = len(counts)
        for idx in idx_to_count:
            if 0 <= idx < total:
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
