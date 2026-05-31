"""Island structure for FLL Scheduler GA."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fll_scheduler_ga.domain.schedule import Schedule
    from fll_scheduler_ga.domain.schema import GaParameterModel
    from fll_scheduler_ga.genetic.context import GaContext
    from fll_scheduler_ga.genetic.stagnation import FitnessHistory, OperatorStats, StagnationHandler
    from fll_scheduler_ga.operators.crossover import Crossover
    from fll_scheduler_ga.operators.mutation import Mutation

logger = getLogger(__name__)


@dataclass(slots=True)
class SchedulePopulation:
    """Population of schedules in the genetic algorithm."""

    schedules: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        """Return the number of schedules in the population."""
        return self.schedules.shape[0] if self.schedules is not None else 0

    def add(self, schedule: np.ndarray) -> None:
        """Add a new schedule to the population."""
        if self.schedules.size == 0:
            self.schedules = schedule[np.newaxis, :]
        else:
            self.schedules = np.vstack((self.schedules, schedule[np.newaxis, :]))


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    context: GaContext
    rng: np.random.Generator
    ga_param: GaParameterModel
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    stagnation: StagnationHandler
    population: SchedulePopulation = field(default_factory=SchedulePopulation)
    selected: list[Schedule] = field(default_factory=list)
    generation: int = 0
    curr_schedule_fits: np.ndarray = field(init=False)

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.population)

    @property
    def n_needed(self) -> int:
        """Return the number of individuals needed to fill the population."""
        return self.ga_param.population_size - len(self)

    def initialize(self) -> None:
        """Initialize the population for each island."""
        if self.n_needed <= 0:
            logger.debug("Island %d: Population already full with %d individuals", self.identity, len(self))
            return
        logger.debug("Island %d: Initializing population with %d individuals", self.identity, self.n_needed)
        self._build_n_schedules(self.n_needed)

    def run_epoch(self) -> None:
        """Run a full epoch: evaluate, select, evolve, and handle stagnation."""
        self._handle_underpopulation()
        self._evolve()
        self._select_next_generation()
        self.fitness_history.update_fitness_history()
        self._check_stagnation()
        self._handle_underpopulation()
        self.generation += 1

    def _handle_underpopulation(self) -> None:
        """Handle underpopulation by creating new individuals."""
        if self.n_needed <= 0:
            return
        logger.debug("Island %d: Handling underpopulation with %d individuals", self.identity, self.n_needed)
        self._build_n_schedules(self.n_needed)

    def _check_stagnation(self) -> None:
        """Check for stagnation without modifying the population."""
        if self.stagnation.is_stagnant(self.generation, self.fitness_history.history):
            # Get the index of the schedule with the best fitness
            sum_fits = self.curr_schedule_fits.sum(axis=1)
            max_idx = sum_fits.argmax()
            # 10% chance to remove the best schedule to encourage diversity
            if self.rng.random() < 0.1:
                idx_to_pop = max_idx
            else:
                candidates = np.delete(np.arange(len(self.selected)), max_idx)
                idx_to_pop = self.rng.choice(candidates)
            self.selected.pop(idx_to_pop)
            self.population.schedules = np.delete(arr=self.population.schedules, obj=idx_to_pop, axis=0)
            logger.debug(
                "Stagnation. Island: %d. Generation: %d. Schedule Removed: %d.",
                self.identity,
                self.generation + 1,
                idx_to_pop,
            )

    def add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        self.operator_stats.count_offspring("total")
        if self.context.check(schedule) and schedule not in self.selected:
            self.selected.append(schedule)
            self.operator_stats.count_offspring("success")
            return True
        return False

    def _build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        created = 0
        while created < needed:
            s = self.context.build()
            if self.context.repair(s) and self.add_to_population(s):
                self.population.add(s.schedule)
                created += 1

    def _get_operator(self, ops: tuple) -> Any:
        """Randomly select an operator from a tuple of operators."""
        return ops[self.rng.integers(0, len(ops))]

    def _mutate_child(self, schedule: Schedule) -> bool:
        """Mutate a child schedule."""
        m: Mutation = self._get_operator(self.context.mutations)
        m_str = str(m)
        self.operator_stats.count_mutation("total", m_str)
        if m.mutate(schedule):
            self.operator_stats.count_mutation("success", m_str)
            schedule.mutations += 1
            return True
        return False

    def _crossover_parents(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        c: Crossover = self._get_operator(self.context.crossovers)
        c_str = str(c)
        for child in c.cross(parents):
            self.operator_stats.count_crossover("total", c_str)
            if self.context.check(child):
                self.operator_stats.count_crossover("success", c_str)
            if self.context.repair(child):
                yield child

    def _evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return
        created_cycle = 0
        while created_cycle < self.ga_param.offspring_size:
            parents_indices = self.context.select_parents(n=len(pop), k=2)
            parents: Iterator[Schedule] = (pop[i] for i in parents_indices)
            c_roll = self.ga_param.crossover_chance > self.rng.random()
            if c_roll and len(self.context.crossovers) > 0:
                offspring = self._crossover_parents(parents)
            else:
                offspring = (p.clone() for p in parents)
            for child in offspring:
                if created_cycle >= self.ga_param.offspring_size:
                    break
                if len(self.context.mutations) > 0:
                    m_roll = True if not c_roll else self.ga_param.mutation_chance > self.rng.random()
                    if m_roll:
                        self._mutate_child(child)
                if self.add_to_population(child):
                    self.population.add(child.schedule)
                created_cycle += 1
        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def _select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        n_pop = self.ga_param.population_size
        schedule_fits, _ = self.context.evaluate(self.population.schedules)
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
        for _ in range(self.ga_param.migration_size):
            i = self.rng.integers(low=0, high=len(self.selected))
            self.population.schedules = np.delete(self.population.schedules, i, axis=0)
            yield self.selected.pop(i)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            if self.add_to_population(schedule=migrant):
                self.population.add(migrant.schedule)
