"""Tools for Non-dominated Sorting Genetic Algorithm II (NSGA-II)."""

from collections import defaultdict

from ..genetic.schedule import Population, Schedule


def non_dominated_sort(population: Population) -> None:
    """Perform non-dominated sorting on a population."""
    dominates: dict[int, Population] = defaultdict(list)
    dominated_by_count: dict[int, int] = defaultdict(int)
    fronts = [[]]
    population_map = {id(p): p for p in population}

    for p_id, p in population_map.items():
        for q_id, q in population_map.items():
            if p_id == q_id:
                continue
            if _dominates(p, q):
                dominates[p_id].append(q)
            elif _dominates(q, p):
                dominated_by_count[p_id] += 1

        if dominated_by_count[p_id] == 0:
            p.rank = 0
            fronts[0].append(p)

    _compute_crowding_dist(fronts[0])

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[id(p)]:
                dominated_by_count[id(q)] -= 1
                if dominated_by_count[id(q)] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            _compute_crowding_dist(next_front)
            fronts.append(next_front)


def _compute_crowding_dist(front: Population) -> None:
    """Calculate the crowding distance for each individual in a front."""
    if not front or not front[0].fitness:
        return

    num_objectives = len(front[0].fitness)
    for p in front:
        p.crowding_distance = 0.0

    for m in range(num_objectives):
        front.sort(key=lambda p: p.fitness[m])

        front[0].crowding_distance = front[-1].crowding_distance = float("inf")

        f_max = front[-1].fitness[m]
        f_min = front[0].fitness[m]
        if f_max == f_min:
            continue

        for i in range(1, len(front) - 1):
            distance = front[i + 1].fitness[m] - front[i - 1].fitness[m]
            front[i].crowding_distance += distance / (f_max - f_min)


def _dominates(p: Schedule, q: Schedule) -> bool:
    """Return True if individual p dominates q."""
    p_scores = p.fitness
    q_scores = q.fitness
    if p_scores is None or q_scores is None:
        return False

    dominates = False
    for ps, qs in zip(p_scores, q_scores, strict=True):
        if ps < qs:
            return False
        if ps > qs:
            dominates = True
    return dominates
