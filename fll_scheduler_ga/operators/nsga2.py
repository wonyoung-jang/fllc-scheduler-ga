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

        if fronts[0]:
            NSGA2.compute_crowding_dist([pop[i] for i in fronts[0]])

        current_front = 0

        while current_front < len(fronts) and fronts[current_front]:
            next_front = []

            for i in fronts[current_front]:
                for j in dominates[i]:
                    dominated_count[j] -= 1

                    if dominated_count[j] == 0:
                        pop[j].rank = current_front + 1
                        next_front.append(j)

            if next_front:
                NSGA2.compute_crowding_dist([pop[i] for i in next_front])
                fronts.append(next_front)

            current_front += 1

    @staticmethod
    def compute_crowding_dist(front: Population) -> None:
        """Calculate the crowding distance for each individual in a front."""
        if not front:
            return

        first_of_front: Schedule = front[0]
        if not first_of_front.fitness:
            return

        num_objectives = len(first_of_front.fitness)
        front_size = len(front)

        for p in front:
            p.crowding = 0.0

        if front_size <= 2:
            for p in front:
                p.crowding = float("inf")
            return

        for m in range(num_objectives):
            front.sort(key=lambda p: p.fitness[m])
            front_min = front[0]
            front_max = front[-1]
            front_min.crowding = float("inf")
            front_max.crowding = float("inf")
            f_min = front_min.fitness[m]
            f_max = front_max.fitness[m]
            f_diff = f_max - f_min

            if f_diff == 0:
                f_diff = 1e-16

            for i in range(1, front_size - 1):
                next_fitness = front[i + 1].fitness[m]
                prev_fitness = front[i - 1].fitness[m]
                diff_fitness = next_fitness - prev_fitness
                front[i].crowding += diff_fitness / f_diff

    @staticmethod
    def dominates(p: Schedule, q: Schedule) -> bool:
        """Check if schedule p dominates schedule q.

        Args:
            p (Schedule): The first schedule to compare.
            q (Schedule): The second schedule to compare.

        Returns:
            bool: True if p dominates q, False otherwise.

        """
        if p.fitness is None or q.fitness is None:
            return False

        better_in_any = False

        for ps, qs in zip(p.fitness, q.fitness, strict=True):
            if ps < qs:
                return False

            if ps > qs:
                better_in_any = True

        return better_in_any
