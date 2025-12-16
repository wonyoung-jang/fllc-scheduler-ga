"""Island structure for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.schemas import GeneticModel
    from ..data_model.schedule import Schedule
    from .builder import ScheduleBuilder
    from .ga_context import GaContext
    from .ga_generation import GaGeneration
    from .population import SchedulePopulation
    from .stagnation import FitnessHistory, OperatorStats, StagnationHandler

logger = getLogger(__name__)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    generation: GaGeneration
    context: GaContext
    rng: np.random.Generator
    genetic_model: GeneticModel
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    builder: ScheduleBuilder
    population: SchedulePopulation

    stagnation: StagnationHandler = field(init=False)
    curr_schedule_fits: np.ndarray = field(init=False)
    selected: list[Schedule] = field(default_factory=list)

    _n_crossovers: int = field(init=False)
    _n_mutations: int = field(init=False)
    _n_pop: int = field(init=False)
    _n_offspring: int = field(init=False)
    _n_migration: int = field(init=False)
    _chance_crossover: float = field(init=False)
    _chance_mutation: float = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up stagnation handler and mutation count."""
        ctx = self.context
        self._n_mutations = len(ctx.mutations)
        self._n_crossovers = len(ctx.crossovers)

        params = self.genetic_model.parameters
        self._n_pop = params.population_size
        self._n_offspring = params.offspring_size
        self._n_migration = params.migration_size
        self._chance_crossover = params.crossover_chance
        self._chance_mutation = params.mutation_chance

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.population)

    def run_epoch(self) -> None:
        """Run a full epoch: evaluate, select, evolve, and handle stagnation."""
        self.handle_underpopulation()
        self.evolve()
        self.select_next_generation()
        self.fitness_history.update_fitness_history()
        if self.stagnation.is_stagnant():
            # Get the index of the schedule with the best fitness
            sum_fits = self.curr_schedule_fits.sum(axis=1)
            max_idx = int(np.argmax(sum_fits))
            # 10% chance to remove the best schedule to encourage diversity
            if self.rng.random() < 0.1:
                idx_to_pop = max_idx
            else:
                non_max_indices = [i for i in range(len(self.selected)) if i != max_idx]
                i = self.rng.integers(0, len(non_max_indices))
                idx_to_pop = non_max_indices[i]

            self.selected.pop(idx_to_pop)
            self.population.schedules = np.delete(arr=self.population.schedules, obj=idx_to_pop, axis=0)
            logger.debug(
                "Stagnation. Island: %d. Generation: %d. Schedule Removed: %d.",
                self.identity,
                self.generation.curr,
                idx_to_pop,
            )
        self.handle_underpopulation()

    def add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        self.operator_stats.count_offspring("total")
        if self.context.check(schedule) and schedule not in self.selected:
            self.selected.append(schedule)
            self.operator_stats.count_offspring("success")
            return True
        return False

    def build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        if needed <= 0:
            return

        created = 0
        while created < needed:
            s = self.builder.build()
            if self.context.repair(s) and self.add_to_population(s):
                self.population.add(s.schedule)
                created += 1

    @property
    def n_needed(self) -> int:
        """Return the number of individuals needed to fill the population."""
        return self._n_pop - len(self)

    def initialize(self) -> None:
        """Initialize the population for each island."""
        need = self.n_needed
        if need == 0:
            logger.debug("Island %d: Population already full with %d individuals", self.identity, len(self))
            return
        logger.debug("Island %d: Initializing population with %d individuals", self.identity, need)
        self.build_n_schedules(need)

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by creating new individuals."""
        need = self.n_needed
        if need == 0:
            return
        logger.debug("Island %d: Handling underpopulation with %d individuals", self.identity, need)
        self.build_n_schedules(need)

    def mutate_schedule(self, schedule: Schedule) -> bool:
        """Mutate a child schedule."""
        m_idx = self.rng.integers(0, self._n_mutations)
        m = self.context.mutations[m_idx]
        m_str = str(m)
        self.operator_stats.count_mutation("total", m_str)
        if m.mutate(schedule):
            self.operator_stats.count_mutation("success", m_str)
            schedule.mutations += 1
            return True
        return False

    def crossover_schedule(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        c_idx = self.rng.integers(0, self._n_crossovers)
        c = self.context.crossovers[c_idx]
        c_str = str(c)
        for child in c.cross(parents):
            self.operator_stats.count_crossover("total", c_str)
            if self.context.check(child):
                self.operator_stats.count_crossover("success", c_str)
            if self.context.repair(child):
                yield child

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return

        created_cycle = 0
        while created_cycle < self._n_offspring:
            parents_indices = self.context.select_parents(n=len(pop), k=2)
            parents = (pop[i] for i in parents_indices)
            c_roll = self._chance_crossover > self.rng.random()
            if c_roll and self._n_crossovers > 0:
                offspring = self.crossover_schedule(parents)
            else:
                offspring = (p.clone() for p in parents)

            for child in offspring:
                if self._n_mutations > 0:
                    m_roll = True if not c_roll else self._chance_mutation > self.rng.random()
                    if m_roll:
                        self.mutate_schedule(child)

                if self.add_to_population(child):
                    self.population.add(child.schedule)

                created_cycle += 1
                if created_cycle >= self._n_offspring:
                    break

        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def evaluate_pop(self) -> tuple[np.ndarray, ...]:
        """Evaluate the entire population."""
        return self.context.evaluate(self.population.schedules)

    def select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        n_pop = self._n_pop
        schedule_fits, _ = self.evaluate_pop()
        if schedule_fits.shape[0] != n_pop:
            _, flat, _ = self.context.select_nsga3(schedule_fits, n_pop)
        else:
            flat = np.arange(n_pop)

        self.fitness_history.current = schedule_fits[flat].mean(axis=0)

        total_pop: list[Schedule] = self.selected
        self.selected = []
        for i in flat:
            self.add_to_population(total_pop[i])

        self.population.schedules = self.population.schedules[flat]
        self.curr_schedule_fits = schedule_fits[flat]

    def give_migrants(self) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        for _ in range(self._n_migration):
            i = self.rng.integers(low=0, high=len(self.selected))
            self.population.schedules = np.delete(self.population.schedules, i, axis=0)
            yield self.selected.pop(i)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            if self.add_to_population(schedule=migrant):
                self.population.add(migrant.schedule)
