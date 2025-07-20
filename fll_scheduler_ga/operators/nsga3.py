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
        k = self.population_size - len(next_pop)

        if len(last_front) < k:
            next_pop.extend(last_front)
            return next_pop

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
        """Select k individuals from the last front using a robust niching mechanism."""
        self._normalize_objectives(fronts_to_consider)
        self._associate_and_calculate_distances()

        ro = self._count_members_per_ref_point([p for front in fronts_to_consider[:-1] for p in front])
        selected: list[Schedule] = []
        member_pool = last_front[:]

        while len(selected) < k:
            if not member_pool:
                break

            min_ro_val = min(ro)
            potential_z_indices = [i for i, val in enumerate(ro) if val == min_ro_val]
            self.rng.shuffle(potential_z_indices)

            chosen_member = None
            z_to_increment = -1

            for z_idx in potential_z_indices:
                associated_members = [p for p in member_pool if p.ref_point_idx == z_idx]
                if associated_members:
                    if ro[z_idx] == 0:
                        chosen_member = min(associated_members, key=lambda p: p.distance_to_ref_point)
                    else:
                        chosen_member = self.rng.choice(associated_members)
                    z_to_increment = z_idx
                    break

            if chosen_member is None:
                chosen_member: Schedule = self.rng.choice(member_pool)
                z_to_increment = chosen_member.ref_point_idx

            selected.append(chosen_member)
            member_pool.remove(chosen_member)
            ro[z_to_increment] += 1

        return selected

    def _normalize_objectives(self, fronts: list[Population]) -> None:
        """Normalize objectives for the entire population being considered."""
        population = [p for front in fronts for p in front]
        if not population or not any(p.fitness for p in population):
            return

        obj_values = np.array([p.fitness for p in population])
        ideal_point = np.min(obj_values, axis=0)
        max_values = np.max(obj_values, axis=0)

        # Calculate nadir point by taking the max from each objective across the last front considered
        last_front_pop = [p for front in fronts for p in front if p.rank == fronts[-1][0].rank]
        nadir_point = np.max(np.array([p.fitness for p in last_front_pop]), axis=0) if last_front_pop else max_values

        range_vals = nadir_point - ideal_point
        range_vals[range_vals < 1e-6] = 1e-6  # Avoid division by zero

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

    def _count_members_per_ref_point(self, population: Population) -> list[int]:
        """Count how many individuals are associated with each reference point."""
        ro = [0] * len(self.ref_points)
        for p in population:
            if p.ref_point_idx is not None:
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
