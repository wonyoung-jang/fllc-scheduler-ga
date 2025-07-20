"""Tools for Non-dominated Sorting Genetic Algorithm III (NSGA-III)."""

import random
from collections import defaultdict
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
    _pop: Population = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to generate reference points."""
        self.ref_points = self.generate_reference_points(self.get_num_divisions())

    def get_num_divisions(self) -> int:
        """Calculate the number of divisions for reference point generation."""
        p = 1
        while (
            len(list(combinations(range(p + self.num_objectives - 1), self.num_objectives - 1))) < self.population_size
        ):
            p += 1
        return p

    def generate_reference_points(self, p: int) -> np.ndarray:
        """Generate a set of structured reference points."""
        points = []
        for c in combinations(range(p + self.num_objectives - 1), self.num_objectives - 1):
            temp_point = [c[0] - 0]
            for i in range(1, self.num_objectives - 1):
                diff = c[i] - c[i - 1]
                temp_point.append(diff)
            temp_point.append(p + self.num_objectives - 1 - c[-1] - 1)
            points.append([x / p for x in temp_point])
        return np.array(points)

    def select(self, population: Population) -> Population:
        """Select the next generation using NSGA-III principles."""
        self._pop = population
        self._non_dominated_sort()
        fronts = self._get_fronts()
        last_front_idx = self._determine_last_front(fronts)

        next_pop = [p for i in range(last_front_idx) for p in fronts[i]]
        if len(next_pop) == self.population_size:
            return next_pop

        last_front = fronts[last_front_idx]
        if not last_front:
            return next_pop

        k = self.population_size - len(next_pop)
        next_pop.extend(self._niching_selection(fronts[: last_front_idx + 1], last_front, k))
        return next_pop

    def _get_fronts(self) -> list[Population]:
        """Group population into fronts."""
        fronts = defaultdict(list)
        for p in self._pop:
            fronts[p.rank].append(p)
        return [fronts[i] for i in sorted(fronts.keys())]

    def _determine_last_front(self, fronts: list[Population]) -> int:
        """Determine which front is the last to be included."""
        count = 0
        for i, f in enumerate(fronts):
            count += len(f)
            if count >= self.population_size:
                return i
        return len(fronts) - 1

    def _non_dominated_sort(self) -> None:
        """Perform non-dominated sorting on the population."""
        pop_size = len(self._pop)
        dominates_list = [[] for _ in range(pop_size)]
        dominated_counts = [0] * pop_size
        ranks = [-1] * pop_size
        fronts = [[]]

        for i, j in combinations(range(pop_size), 2):
            p, q = self._pop[i], self._pop[j]
            if self.dominates(p, q):
                dominates_list[i].append(j)
                dominated_counts[j] += 1
            elif self.dominates(q, p):
                dominates_list[j].append(i)
                dominated_counts[i] += 1

        for i in range(pop_size):
            if dominated_counts[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        current_front_idx = 0
        while current_front_idx < len(fronts) and fronts[current_front_idx]:
            next_front = []
            for i in fronts[current_front_idx]:
                for j in dominates_list[i]:
                    dominated_counts[j] -= 1
                    if dominated_counts[j] == 0:
                        ranks[j] = current_front_idx + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front_idx += 1

        for i in range(pop_size):
            self._pop[i].rank = ranks[i]

    def _niching_selection(self, fronts_to_consider: list[Population], last_front: Population, k: int) -> Population:
        """Select k individuals from the last front using niching."""
        self._normalize_objectives(fronts_to_consider)
        self._associate_and_calculate_distances()

        ro = self._count_members_per_ref_point(fronts_to_consider[:-1])
        selected_indices: set[int] = set()
        member_pool = last_front[:]

        while len(selected_indices) < k:
            min_ro = min(ro)
            min_ro_indices = [i for i, x in enumerate(ro) if x == min_ro]
            self.rng.shuffle(min_ro_indices)
            found = False
            for z_idx in min_ro_indices:
                members_of_z = [p for p in member_pool if p.ref_point_idx == z_idx]
                if not members_of_z:
                    continue

                if ro[z_idx] == 0:
                    best_member = min(members_of_z, key=lambda p: p.distance_to_ref_point)
                else:
                    best_member = self.rng.choice(members_of_z)

                selected_indices.add(self._pop.index(best_member))
                member_pool.remove(best_member)
                ro[z_idx] += 1
                found = True
                break
            if not found:
                break
        return [self._pop[i] for i in selected_indices]

    def _normalize_objectives(self, fronts: list[Population]) -> None:
        """Normalize objectives for the entire population being considered."""
        population = [p for front in fronts for p in front]
        if not population:
            return

        obj_values = np.array([p.fitness for p in population])
        ideal_point = np.min(obj_values, axis=0)
        max_values = np.max(obj_values, axis=0)
        range_vals = max_values - ideal_point
        range_vals[range_vals == 0] = 1e-6
        normalized = (obj_values - ideal_point) / range_vals

        for p, norm_fit in zip(population, normalized, strict=False):
            p.normalized_fitness = norm_fit

    def _associate_and_calculate_distances(self) -> None:
        """Associate individuals with the nearest reference points and store distances."""
        for p in self._pop:
            if p.normalized_fitness is None:
                continue
            distances = np.sum((p.normalized_fitness - self.ref_points) ** 2, axis=1)
            p.ref_point_idx = np.argmin(distances)
            p.distance_to_ref_point = distances[p.ref_point_idx]

    def _count_members_per_ref_point(self, fronts: list[Population]) -> list[int]:
        """Count how many individuals from non-last fronts are associated with each reference point."""
        ro = [0] * len(self.ref_points)
        for front in fronts:
            for p in front:
                ro[p.ref_point_idx] += 1
        return ro

    @staticmethod
    def dominates(p: Schedule, q: Schedule) -> bool:
        """Check if schedule p dominates schedule q."""
        if p.fitness is None or q.fitness is None:
            return False
        better_in_any = False
        for ps, qs in zip(p.fitness, q.fitness, strict=True):
            if ps < qs:
                return False
            if ps > qs:
                better_in_any = True
        return better_in_any
