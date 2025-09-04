"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from itertools import chain, combinations
from logging import getLogger
from math import comb
from random import Random
from typing import Any

import numpy as np

from ..data_model.schedule import Population, Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: Random
    num_objectives: int
    population_size: int
    ref_points: np.ndarray = field(default=None, repr=False)

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
        m = self.num_objectives
        if not (pts := list(self._gen_ref_points())):
            pts = [[1.0 / m] * m]
        self.ref_points = np.asarray(pts, dtype=float)
        logger.debug("Generated %d reference points:\n%s", len(self.ref_points), self.ref_points)

    def select(self, population: Population | Any, population_size: int) -> dict[int, Schedule]:
        """Select the next generation using NSGA-III principles."""
        if not isinstance(population, list):
            population = list(population)

        fronts = NSGA3._non_dominated_sort(population, population_size)
        last_front = fronts[-1] if fronts else []
        self._normalize_then_associate(fronts, last_front)
        if len(fronts) == 1:
            selected = list(chain.from_iterable(fronts))[:population_size]
            return {hash(p): p for p in selected}

        selected = list(chain.from_iterable(fronts[:-1]))
        selected.extend(
            self._niching(
                fronts=fronts,
                last_front=last_front,
                k=population_size - len(selected),
            )
        )
        return {hash(p): p for p in selected}

    @staticmethod
    def _non_dominated_sort(pop: Population, pop_size: int) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        n = len(pop)
        dom_list: list[list[int]] = [[] for _ in range(n)]
        dom_count = [0] * n
        fits = [p.fitness for p in pop]

        for i, fi in enumerate(fits):
            for j in range(i + 1, n):
                fj = fits[j]
                if dominates(fi, fj):
                    dom_list[i].append(j)
                    dom_count[j] += 1
                elif dominates(fj, fi):
                    dom_list[j].append(i)
                    dom_count[i] += 1

        fronts: list[list[int]] = [[]]
        for i in range(n):
            if dom_count[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)

        pop_count = len(fronts[0])
        curr = 0
        while curr < len(fronts) and fronts[curr] and pop_count < pop_size:
            next_front: list[int] = []
            for i in fronts[curr]:
                for j in dom_list[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        pop[j].rank = curr + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                pop_count += len(next_front)
            curr += 1

        return [[pop[i] for i in fr] for fr in fronts]

    def _niching(self, fronts: list[Population], last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        counts = self._count(p.ref_point for fr in fronts[:-1] for p in fr)
        pool: dict[int, Schedule] = dict(enumerate(last_front))
        selected = 0

        while selected < k and pool:
            pool_ref_points = {s.ref_point for s in pool.values()}
            next_counts = [(i, counts[i]) for i in pool_ref_points]
            if not next_counts:
                break

            min_count = min(c for _, c in next_counts)
            min_refs = [i for i, c in next_counts if c == min_count]
            self.rng.shuffle(min_refs)

            for niche in min_refs:
                cluster = sorted(
                    ((idx, s) for idx, s in pool.items() if s.ref_point == niche),
                    key=lambda idx_s: idx_s[1].ref_distance,
                )
                if not cluster:
                    continue

                pick_idx, _ = cluster[0] if counts[niche] == 0 else self.rng.choice(cluster)
                yield pool.pop(pick_idx)
                counts[niche] += 1
                selected += 1
                if selected >= k:
                    break

    def _normalize_then_associate(self, fronts: list[Population], last_front: Population) -> None:
        """Normalize objectives then associate individuals with nearest reference points."""
        all_schedules = list(chain.from_iterable(fronts))
        if not all_schedules:
            return

        fits_all = np.asarray([p.fitness for p in all_schedules], dtype=float)
        fits_last = np.asarray([p.fitness for p in last_front], dtype=float)

        # Compute ideal (max) and nadir (min on the last front) and normalize
        epsilon = 1e-12
        ideal = fits_all.max(axis=0)
        nadir = fits_last.min(axis=0)
        span = ideal - nadir
        span[span == 0] = epsilon
        norm_fits: np.ndarray = (fits_all - nadir) / span

        ref_pts = self.ref_points
        w_norm_sq = np.sum(ref_pts**2, axis=1)  # (H,)
        w_norm_sq[w_norm_sq == 0] = epsilon

        dots = norm_fits @ ref_pts.T
        coeffs = dots / w_norm_sq
        proj = coeffs[:, :, None] * ref_pts[None, :, :]
        f_expanded = norm_fits[:, None, :]
        residuals = f_expanded - proj
        dists = np.linalg.norm(residuals, axis=2)
        min_vals = dists.min(axis=1)
        tie_mask = dists == min_vals[:, None]

        for i, s in enumerate(all_schedules):
            candidates = np.nonzero(tie_mask[i])[0]
            s.ref_point = int(self.rng.choice(candidates))
            s.ref_distance = float(min_vals[i])

    def _count(self, idx_to_count: Iterator[int]) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        counts = [0] * len(self.ref_points)
        for idx in idx_to_count:
            if 0 <= idx < len(counts):
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
