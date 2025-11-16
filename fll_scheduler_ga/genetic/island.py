"""Island structure for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from .stagnation import FitnessHistory

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterator

    from ..config.schemas import GaParameters
    from ..data_model.schedule import Schedule
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from .builder import ScheduleBuilder
    from .ga_context import GaContext

logger = getLogger(__name__)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    context: GaContext
    rng: np.random.Generator
    ga_params: GaParameters
    offspring_ratio: Counter
    crossover_ratio: dict[str, Counter]
    mutation_ratio: dict[str, Counter]

    fitness_history: FitnessHistory = None

    builder: ScheduleBuilder = None
    selected: list[Schedule] = field(default_factory=list, repr=False)
    curr_gen: int = 0

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.builder = self.context.builder
        n_gen = self.ga_params.generations
        n_obj = len(self.context.evaluator.objectives)

        self.fitness_history = FitnessHistory(
            curr_gen=self.curr_gen,
            curr_fit=np.zeros((1, n_obj), dtype=float),
            history=np.zeros((n_gen, n_obj), dtype=float),
        )

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [s for s in self.selected if s.rank == 0]

    def add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        self.offspring_ratio["total"] += 1
        if schedule not in self.selected:
            self.selected.append(schedule)
            self.offspring_ratio["success"] += 1
            return True
        return False

    def build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        if needed <= 0:
            return

        created = 0
        while created < needed:
            s = self.builder.build()
            if self.context.repairer.repair(s) and self.add_to_population(s):
                created += 1

    def initialize(self) -> None:
        """Initialize the population for each island."""
        needed = self.ga_params.population_size - len(self)
        if needed == 0:
            logger.debug("Island %d: Population already full with %d individuals", self.identity, len(self))
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

    def mutate_schedule(self, schedule: Schedule) -> bool:
        """Mutate a child schedule."""
        m: Mutation = self.rng.choice(self.context.mutations)
        m_str = str(m)
        self.mutation_ratio["total"][m_str] += 1
        if m.mutate(schedule):
            self.mutation_ratio["success"][m_str] += 1
            schedule.mutations += 1
            return True
        return False

    def crossover_schedule(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        c: Crossover = self.rng.choice(self.context.crossovers)
        c_str = str(c)
        for child in c.cross(parents):
            self.crossover_ratio["total"][c_str] += 1
            if self.context.checker.check(child):
                self.crossover_ratio["success"][c_str] += 1
            if self.context.repairer.repair(child):
                yield child

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return

        created_cycle = 0
        while created_cycle < self.ga_params.offspring_size:
            parents_indices = self.context.selection.select(len(pop), k=2)
            parents = (pop[i] for i in parents_indices)
            c_roll = self.ga_params.crossover_chance > self.rng.random()
            if c_roll and self.context.crossovers:
                offspring = self.crossover_schedule(parents)
            else:
                offspring = (p.clone() for p in parents)

            for child in offspring:
                m_roll = self.ga_params.mutation_chance > self.rng.random()
                if m_roll and self.context.mutations:
                    self.mutate_schedule(child)
                if self.context.checker.check(child):
                    self.add_to_population(child)

                created_cycle += 1
                if created_cycle >= self.ga_params.offspring_size:
                    break

        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def evaluate_pop(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the entire population."""
        pop_array = np.array([s.schedule for s in self.selected], dtype=int)
        return self.context.evaluator.evaluate_population(pop_array)

    def select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        n_pop = self.ga_params.population_size
        schedule_fits, _ = self.evaluate_pop()
        fronts = self.context.nsga3.select(schedule_fits, n_pop)
        idx_to_select = [i for f in fronts for i in f]

        curr_fit = schedule_fits[idx_to_select].mean(axis=0)
        self.fitness_history.curr_fit = curr_fit

        total_pop: list[Schedule] = self.selected

        self.selected = []
        for i in idx_to_select:
            self.add_to_population(total_pop[i])

    def give_migrants(self) -> list[Schedule]:
        """Randomly yield migrants from population."""
        migrants = []
        for _ in range(self.ga_params.migration_size):
            i = self.rng.integers(0, len(self.selected))
            migrants.append(self.selected.pop(i))
        return migrants

    def receive_migrants(self, migrants: list[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)
