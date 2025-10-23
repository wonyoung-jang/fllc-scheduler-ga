"""Island structure for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterator

    from ..config.ga_context import GaContext
    from ..config.ga_parameters import GaParameters
    from ..data_model.schedule import Schedule
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from .builder import ScheduleBuilder

logger = getLogger(__name__)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    context: GaContext
    offspring_ratio: Counter
    crossover_ratio: dict[str, Counter]
    mutation_ratio: dict[str, Counter]

    rng: np.random.Generator = None
    builder: ScheduleBuilder = None
    ga_params: GaParameters = None

    selected: list[Schedule] = field(default_factory=list, repr=False)

    curr_gen: int = 0
    fitness_history: np.ndarray = None
    curr_schedule_fitnesses: np.ndarray = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.rng = self.context.app_config.rng
        self.builder = self.context.builder
        self.ga_params = self.context.app_config.ga_params
        n_gen = self.ga_params.generations
        n_obj = len(self.context.evaluator.objectives)
        self.fitness_history = np.zeros((n_gen, n_obj), dtype=float)
        self.curr_schedule_fitnesses = np.zeros((0, n_obj), dtype=float)

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[self.curr_gen - 1] if self.curr_gen > 0 else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        if self.curr_schedule_fitnesses is not None and self.curr_schedule_fitnesses.size > 0:
            self.fitness_history[self.curr_gen] = self.curr_schedule_fitnesses.mean(axis=0)

    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [s for s in self.selected if s.rank == 0]

    def add_to_population(self, schedule: Schedule, *, recurse: bool = False) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        if not recurse:
            self.offspring_ratio["total"] += 1

        if schedule not in self.selected:
            self.selected.append(schedule)
            self.offspring_ratio["success"] += 1
            return True

        if not self.context.mutations:
            return False

        self.mutate_schedule(schedule)
        return self.add_to_population(schedule, recurse=True)

    def build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        if needed <= 0:
            return

        created = 0
        while created < needed:
            s = self.builder.build()
            if self.context.repairer(s):
                self.add_to_population(s)
                created += 1

    def initialize(self) -> None:
        """Initialize the population for each island."""
        needed = self.ga_params.population_size - len(self)
        if needed == 0:
            return
        logger.debug("Island %d: Initializing population with %d individuals", self.identity, needed)
        self.build_n_schedules(needed)

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by creating new individuals."""
        needed = self.ga_params.population_size - len(self)
        if needed == 0:
            return
        logger.debug("Island %d: Handling underpopulation with %d individuals", self.identity, needed)
        self.build_n_schedules(needed)

    def mutate_schedule(self, schedule: Schedule, *, m_roll: bool = True) -> bool:
        """Mutate a child schedule."""
        if not m_roll or not self.context.mutations:
            return False

        m: Mutation = self.rng.choice(self.context.mutations)

        m_str = str(m)
        self.mutation_ratio["total"][m_str] += 1
        if m.mutate(schedule):
            self.mutation_ratio["success"][m_str] += 1
            schedule.mutations += 1
            return True
        return False

    def crossover_schedule(self, parents: Iterator[Schedule], *, c_roll: bool = True) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        if not c_roll or not self.context.crossovers:
            yield from (p.clone() for p in parents)
            return

        c: Crossover = self.rng.choice(self.context.crossovers)

        c_str = str(c)
        for child in c.cross(parents):
            self.crossover_ratio["total"][c_str] += 1
            if self.context.checker(child) and child not in self.selected:
                self.crossover_ratio["success"][c_str] += 1
            if self.context.repairer(child):
                yield child

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return

        created = 0
        while created < self.ga_params.offspring_size:
            parents_indices = self.context.selection.select(len(pop), k=2)
            parents = (pop[i] for i in parents_indices)
            c_roll = self.ga_params.crossover_chance > self.rng.random()
            for child in self.crossover_schedule(parents, c_roll=c_roll):
                m_roll = self.ga_params.mutation_chance > self.rng.random()
                self.mutate_schedule(child, m_roll=m_roll)
                self.add_to_population(child)
                created += 1
                if created >= self.ga_params.offspring_size:
                    break

        if not self.selected:
            return

        n_pop = self.ga_params.population_size
        schedule_fitnesses, _ = self.evaluate_pop()
        pop_to_select: list[Schedule] = self.selected
        fronts = self.context.nsga3.select(
            fits=schedule_fitnesses,
            n=n_pop,
        )
        self.curr_schedule_fitnesses = schedule_fitnesses
        self.selected = []
        for front in fronts:
            for idx in front:
                self.add_to_population(pop_to_select[idx])

    def evaluate_pop(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the entire population."""
        pop_array = np.array([s.schedule for s in self.selected], dtype=int)
        return self.context.evaluator.evaluate_population(pop_array)

    def give_migrants(self) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        for _ in range(self.ga_params.migration_size):
            i = self.rng.integers(0, len(self.selected))
            yield self.selected.pop(i)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)
