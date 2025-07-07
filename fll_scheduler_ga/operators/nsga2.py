"""Tools for Non-dominated Sorting Genetic Algorithm II (NSGA-II)."""

from ..genetic.schedule import Population, Schedule


def non_dominated_sort(population: Population) -> None:
    """Perform non-dominated sorting on a population."""
    n = len(population)
    dominates_ = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = population[i], population[j]
            if _dominates(pi, pj):
                dominates_[i].append(j)
                dominated_count[j] += 1
            elif _dominates(pj, pi):
                dominates_[j].append(i)
                dominated_count[i] += 1

    for i in range(n):
        if dominated_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    _compute_crowding_dist([population[i] for i in fronts[0]])

    current_front = 0

    while current_front < len(fronts):
        next_front = []

        for i in fronts[current_front]:
            for j in dominates_[i]:
                dominated_count[j] -= 1

                if dominated_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)

        current_front += 1

        if next_front:
            _compute_crowding_dist([population[i] for i in next_front])
            fronts.append(next_front)


def _compute_crowding_dist(front: Population) -> None:
    """Calculate the crowding distance for each individual in a front."""
    if not front or not front[0].fitness:
        return

    num_objectives = len(front[0].fitness)

    for p in front:
        p.crowding = 0.0

    for m in range(num_objectives):
        front.sort(key=lambda p: p.fitness[m])
        f_min = front[0].fitness[m]
        f_max = front[-1].fitness[m]
        front[0].crowding = front[-1].crowding = float("inf")

        if f_max == f_min:
            continue

        inv_range = 1.0 / (f_max - f_min)

        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].fitness[m] - front[i - 1].fitness[m]) * inv_range


def _dominates(p: Schedule, q: Schedule) -> bool:
    """Return True if individual p dominates q."""
    if p.fitness is None or q.fitness is None:
        return False

    better_in_any = False

    for ps, qs in zip(p.fitness, q.fitness, strict=True):
        if ps < qs:
            return False

        if ps > qs:
            better_in_any = True

    return better_in_any
