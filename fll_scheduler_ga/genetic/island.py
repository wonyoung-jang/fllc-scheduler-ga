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
        self.operator_stats.offspring["total"] += 1
        if self.context.checker.check(schedule) and schedule not in self.selected:
            self.selected.append(schedule)
            self.operator_stats.offspring["success"] += 1
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
                self.population.add_schedule(s.schedule)
                created += 1

    @property
    def n_needed(self) -> int:
        """Return the number of individuals needed to fill the population."""
        return self.genetic_model.parameters.population_size - len(self)

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
        m_idx = self.rng.integers(0, len(self.context.mutations))
        m = self.context.mutations[m_idx]
        m_str = str(m)
        self.operator_stats.mutation["total"][m_str] += 1
        if m.mutate(schedule):
            self.operator_stats.mutation["success"][m_str] += 1
            schedule.mutations += 1
            return True
        return False

    def crossover_schedule(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        c_idx = self.rng.integers(0, len(self.context.crossovers))
        c = self.context.crossovers[c_idx]
        c_str = str(c)
        for child in c.cross(parents):
            self.operator_stats.crossover["total"][c_str] += 1
            if self.context.checker.check(child):
                self.operator_stats.crossover["success"][c_str] += 1
            if self.context.repairer.repair(child):
                yield child

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return

        created_cycle = 0
        while created_cycle < self.genetic_model.parameters.offspring_size:
            crossovered = False
            parents_indices = self.context.selection.select(len(pop), k=2)
            parents = (pop[i] for i in parents_indices)
            c_roll = self.genetic_model.parameters.crossover_chance > self.rng.random()
            if c_roll and self.context.crossovers:
                offspring = self.crossover_schedule(parents)
                crossovered = True
            else:
                offspring = (p.clone() for p in parents)

            for child in offspring:
                m_roll = self.genetic_model.parameters.mutation_chance > self.rng.random()
                if (m_roll or not crossovered) and self.context.mutations:
                    self.mutate_schedule(child)

                if self.add_to_population(child):
                    self.population.add_schedule(child.schedule)
                created_cycle += 1
                if created_cycle >= self.genetic_model.parameters.offspring_size:
                    break

        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def evaluate_pop(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the entire population."""
        pop_array = self.population.schedules
        return self.context.evaluator.evaluate_population(pop_array)

    def select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        n_pop = self.genetic_model.parameters.population_size
        schedule_fits, _ = self.evaluate_pop()
        if schedule_fits.shape[0] != n_pop:
            _, flat, _ = self.context.nsga3.select(schedule_fits, n_pop)
        else:
            flat = np.arange(n_pop)

        self.fitness_history.current = schedule_fits[flat].mean(axis=0)

        total_pop: list[Schedule] = self.selected
        self.selected = []
        for i in flat:
            self.add_to_population(total_pop[i])

        self.population.schedules = self.population.schedules[flat]
        self.curr_schedule_fits = schedule_fits[flat]

    def give_migrants(self) -> list[Schedule]:
        """Randomly yield migrants from population."""
        migrants = []
        for _ in range(self.genetic_model.parameters.migration_size):
            i = self.rng.integers(low=0, high=len(self.selected))
            migrants.append(self.selected.pop(i))
            self.population.schedules = np.delete(self.population.schedules, i, axis=0)
        return migrants

    def receive_migrants(self, migrants: list[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            if self.add_to_population(schedule=migrant):
                self.population.add_schedule(migrant.schedule)
