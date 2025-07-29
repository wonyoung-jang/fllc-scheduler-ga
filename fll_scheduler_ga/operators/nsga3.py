"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..genetic.schedule import Population, Schedule


@dataclass(slots=True)
class NSGA3:
    """Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

    rng: random.Random
    num_objectives: int
    population_size: int
    ref_points: np.ndarray = field(init=False, repr=False)
    _pop: Population = field(init=False, repr=False)

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

    def select(self, population: Population, population_size: int = 0) -> tuple[Population, set[int]]:
        """Select the next generation using NSGA-III principles."""
        self._pop = population
        pop_size = population_size or self.population_size

        fronts = self._non_dominated_sort()
        last_idx = self._get_last_front_idx(fronts, pop_size)
        selected = [p for i in range(last_idx) for p in fronts[i]]
        last_front = fronts[last_idx]
        k = pop_size - len(selected)

        selected.extend(self._niching(fronts, last_front, k))
        selected_hashes = {hash(p) for p in selected}
        return selected, selected_hashes

    def _get_last_front_idx(self, fronts: list[Population], pop_size: int) -> int:
        """Determine which front is the last to be included."""
        total = 0
        for i, front in enumerate(fronts):
            total += len(front)
            if total >= pop_size:
                return i
        return len(fronts) - 1

    def _non_dominated_sort(self) -> list[Population]:
        """Perform non-dominated sorting on the population."""
        size = len(self._pop)
        dominates_list = [[] for _ in range(size)]
        dominated_counts = [0] * size
        fronts = [[]]

        for i, j in combinations(range(size), 2):
            p, q = self._pop[i], self._pop[j]
            if self._dominates(p.fitness, q.fitness):
                dominates_list[i].append(j)
                dominated_counts[j] += 1
            elif self._dominates(q.fitness, p.fitness):
                dominates_list[j].append(i)
                dominated_counts[i] += 1

        for i in range(size):
            if dominated_counts[i] == 0:
                fronts[0].append(i)

        curr = 0
        while curr < len(fronts) and fronts[curr]:
            next_front = []
            for i in fronts[curr]:
                for j in dominates_list[i]:
                    dominated_counts[j] -= 1
                    if dominated_counts[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            curr += 1

        for rank, front in enumerate(fronts):
            for i in front:
                self._pop[i].rank = rank

        return [[self._pop[i] for i in front] for front in fronts]

    def _niching(self, fronts: list[Population], last_front: Population, k: int) -> Iterator[Schedule]:
        """Select k individuals from the last front using a niching mechanism."""
        all_schedules = [p for front in fronts for p in front]
        self._normalize(all_schedules, fronts[-1])
        self._associate(all_schedules)

        counts: list[int] = self._count(fronts[:-1])
        pool = last_front[:]
        selected = 0

        while selected < k and pool:
            picked = False
            min_count = min(counts)
            dirs = [i for i, c in enumerate(counts) if c == min_count]
            for d in self.rng.sample(dirs, k=len(dirs)):
                if not (cluster := [p for p in pool if p.ref_point_idx == d]):
                    continue

                if counts[d] == 0:
                    pick = min(cluster, key=lambda p: p.distance_to_ref_point)
                else:
                    pick = self.rng.choice(cluster)

                picked = True
                pool.remove(pick)
                counts[d] += 1
                selected += 1
                yield pick

                if selected >= k:
                    break

            if not picked and pool:
                pick = self.rng.choice(pool)
                counts[pick.ref_point_idx] += 1
                pool.remove(pick)
                selected += 1
                yield pick

    def _normalize(self, pop: list[Schedule], last_front: Population) -> None:
        """Normalize objectives for the entire population being considered."""
        all_fitnesses = [p.fitness for p in pop]
        all_fitnesses_last = [p.fitness for p in last_front]

        fits = np.array(all_fitnesses)
        fits_last = np.array(all_fitnesses_last)

        # Calculate nadir point by taking the max from each objective across the last front considered
        ideal = fits.min(axis=0)
        nadir = fits_last.max(axis=0) if last_front else fits.max(axis=0)

        span = nadir - ideal
        span[span == 0] = 1

        for p, fit in zip(pop, fits, strict=True):
            p.normalized_fitness = (fit - ideal) / span

    def _associate(self, pop: list[Schedule]) -> None:
        """Associate individuals with the nearest reference points and store distances."""
        for p in pop:
            if (nf := getattr(p, "normalized_fitness", None)) is None:
                continue
            dists = np.sum((nf - self.ref_points) ** 2, axis=1)
            idx = int(np.argmin(dists))
            p.ref_point_idx = idx
            p.distance_to_ref_point = dists[idx]

    def _count(self, fronts: list[Population]) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        counts = [0] * len(self.ref_points)
        for front in fronts:
            for p in front:
                if (idx := getattr(p, "ref_point_idx", None)) is None:
                    continue
                counts[idx] += 1
        return counts

    def _dominates(self, p_fit: tuple[float] | None, q_fit: tuple[float] | None) -> bool:
        """Check if schedule p dominates schedule q."""
        if p_fit is None or q_fit is None:
            return False

        better_in_any = False
        for ps, qs in zip(p_fit, q_fit, strict=True):
            if ps < qs:
                return False
            if ps > qs:
                better_in_any = True
        return better_in_any
