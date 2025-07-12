"""Tools for Non-dominated Sorting Genetic Algorithm II (NSGA-II)."""

from dataclasses import dataclass

from ..genetic.schedule import Population, Schedule


@dataclass(slots=True)
class NSGA2:
    """Non-dominated Sorting Genetic Algorithm II (NSGA-II)."""

    @staticmethod
    def non_dominated_sort(pop: Population) -> None:
        """Perform non-dominated sorting on a population."""
        pop_size = len(pop)
        dominates: list[list[int]] = [[] for _ in range(pop_size)]
        dominated_count: list[int] = [0] * pop_size
        fronts: list[list[int]] = [[]]

        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if NSGA2.dominates(pop[i], pop[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
                elif NSGA2.dominates(pop[j], pop[i]):
                    dominates[j].append(i)
                    dominated_count[i] += 1

        for i in range(pop_size):
            if dominated_count[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)

        NSGA2.compute_crowding_dist([pop[i] for i in fronts[0]])

        current_front = 0

        while current_front < len(fronts):
            next_front = []

            for i in fronts[current_front]:
                for j in dominates[i]:
                    dominated_count[j] -= 1

                    if dominated_count[j] == 0:
                        pop[j].rank = current_front + 1
                        next_front.append(j)

            current_front += 1

            if next_front:
                NSGA2.compute_crowding_dist([pop[i] for i in next_front])
                fronts.append(next_front)

    @staticmethod
    def compute_crowding_dist(front: Population) -> None:
        """Calculate the crowding distance for each individual in a front."""
        first_of_front: Schedule = front[0]
        if not front or not first_of_front.fitness:
            return

        num_objectives = len(first_of_front.fitness)

        for p in front:
            p.crowding = 0.0

        for m in range(num_objectives):
            front.sort(key=lambda p: p.fitness[m])
            front[0].crowding = float("inf")
            front[-1].crowding = float("inf")

            f_min = front[0].fitness[m]
            f_max = front[-1].fitness[m]
            f_diff = f_max - f_min

            if f_diff == 0:
                continue

            inv_range = 1.0 / f_diff

            for i in range(1, len(front) - 1):
                next_in_front_fitness = front[i + 1].fitness[m]
                prev_in_front_fitness = front[i - 1].fitness[m]
                front[i].crowding += (next_in_front_fitness - prev_in_front_fitness) * inv_range

    @staticmethod
    def dominates(p: Schedule, q: Schedule) -> bool:
        """Check if individual p dominates individual q."""
        if p.fitness is None or q.fitness is None:
            return False

        better_in_any = False

        for ps, qs in zip(p.fitness, q.fitness, strict=True):
            if ps < qs:
                return False

            if ps > qs:
                better_in_any = True

        return better_in_any
