"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from logging import getLogger

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

    def __post_init__(self) -> None:
        """Post-initialization to generate reference points."""
        self._generate_reference_points()

    def _get_num_divisions(self) -> int:
        """Calculate the number of divisions for reference point generation."""
        p = 1
        m = self.num_objectives
        while len(list(combinations(range(p + m - 1), m - 1))) < self.population_size:
            p += 1
        return p

    def _generate_reference_points(self) -> None:
        """Generate a set of structured reference points."""
        p = self._get_num_divisions()
        m = self.num_objectives
        points = []
        for c in combinations(range(p + m - 1), m - 1):
            coords = []
            prev = -1
            for idx in c:
                coords.append(idx - prev - 1)
                prev = idx
            coords.append(p + m - 1 - c[-1] - 1)
            points.append([x / p for x in coords])

        self.ref_points = np.array(points)
        logger.debug("Generated %d reference points:\n%s", len(self.ref_points), self.ref_points)

    def select(self, population: Population | None, population_size: int = 0) -> dict[int, Schedule]:
        """Select the next generation using NSGA-III principles."""
        if not isinstance(population, list):
            population = list(population)

        pop_size = population_size
        fronts = self._non_dominated_sort(population)
        last_idx = self._get_last_front_idx(fronts, pop_size)

        selected = [p for i in range(last_idx) for p in fronts[i]]
        if len(selected) == pop_size:
            return {hash(p): p for p in selected}

        last_front = fronts[last_idx]
        fronts_to_last = fronts[: last_idx + 1]
        k = pop_size - len(selected)
        selected.extend(self._niching(fronts_to_last, last_front, k))
        return {hash(p): p for p in selected}

    def _get_last_front_idx(self, fronts: list[Population], pop_size: int) -> int:
        """Determine which front is the last to be included."""
        total = 0
        for i, front in enumerate(fronts):
            total += len(front)
            if total >= pop_size:
                return i
        return len(fronts) - 1

    def _non_dominated_sort(self, pop: Population) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        size = len(pop)
        dominates_list = [[] for _ in range(size)]
        dominated_counts = [0] * size
        fronts = [[]]

        for i, j in combinations(range(size), 2):
            p_fit, q_fit = pop[i].fitness, pop[j].fitness
            if self._dominates(p_fit, q_fit):
                dominates_list[i].append(j)
                dominated_counts[j] += 1
            elif self._dominates(q_fit, p_fit):
                dominates_list[j].append(i)
                dominated_counts[i] += 1

        curr = 0
        fronts[curr].extend(i for i in range(size) if dominated_counts[i] == 0)

        while curr < len(fronts) and fronts[curr]:
            next_front = []
            for i in fronts[curr]:
                pop[i].rank = curr
                for j in dominates_list[i]:
                    dominated_counts[j] -= 1
                    if dominated_counts[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)

            curr += 1

        return [[pop[i] for i in front] for front in fronts]

    def _niching(self, fronts: list[Population], last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        all_schedules = [p for front in fronts for p in front]
        self._normalize_then_associate(all_schedules, last_front)
        counts = self._count(p.ref_point_idx for fr in fronts[:-1] for p in fr if p.ref_point_idx is not None)
        pool = dict(enumerate(last_front))
        selected = 0

        while selected < k and pool:
            picked = False
            min_count = min(counts)
            min_count_indices = [di for di, c in enumerate(counts) if c == min_count]
            self.rng.shuffle(min_count_indices)
            for d in min_count_indices:
                if not (clst := [(i, p) for i, p in pool.items() if p.ref_point_idx == d]):
                    continue
                pi, _ = min(clst, key=lambda p: p[1].distance_to_ref_point) if counts[d] == 0 else self.rng.choice(clst)
                counts[d] += 1
                selected += 1
                picked = True
                pick = pool.pop(pi)
                yield pick

                if selected >= k:
                    break

            if not picked and pool:
                pi = self.rng.choice(list(pool.keys()))
                pick = pool.pop(pi)
                counts[pick.ref_point_idx] += 1
                selected += 1
                yield pick

    def _normalize_then_associate(self, pop: list[Schedule], last_front: Population) -> None:
        """Normalize objectives then associate individuals with nearest reference points."""
        all_fitnesses = [p.fitness for p in pop]
        all_fitnesses_last = [p.fitness for p in last_front]

        fits = np.array(all_fitnesses)
        fits_last = np.array(all_fitnesses_last)

        # Calculate nadir point by taking the max from each objective across the last front considered
        ideal = fits.min(axis=0)
        nadir = fits_last.max(axis=0) if last_front else fits.max(axis=0)

        span = nadir - ideal
        span[span == 0] = 1e-6

        for p, fit in zip(pop, fits, strict=True):
            p.normalized_fitness = (fit - ideal) / span
            dists = np.sum((p.normalized_fitness - self.ref_points) ** 2, axis=1)
            p.ref_point_idx = np.argmin(dists)
            p.distance_to_ref_point = dists[p.ref_point_idx]

    def _count(self, idx_to_count: Iterator[int]) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        counts = [0] * len(self.ref_points)
        for idx in idx_to_count:
            counts[idx] += 1
        return counts

    def _dominates(self, p_fit: tuple[float, ...], q_fit: tuple[float, ...]) -> bool:
        """Check if schedule p dominates schedule q."""
        better_in_any = False
        for ps, qs in zip(p_fit, q_fit, strict=True):
            if ps < qs:
                return False
            if ps > qs:
                better_in_any = True
        return better_in_any
