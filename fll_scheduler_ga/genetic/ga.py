"""Genetic algorithm for FLL Scheduler GA."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fll_scheduler_ga.domain.model import GaObserver, Schedule
    from fll_scheduler_ga.genetic.context import GaContext
    from fll_scheduler_ga.genetic.operator import Crossover, Mutation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    ctx: GaContext
    obs: tuple[GaObserver, ...]
    opstat: OperatorStats
    fitness_history: FitnessHistory
    gen_arr: np.ndarray
    seed_pop: list[Schedule]
    island_seed_map: dict[int, list[int]]
    islands: list[Island]
    start_time: float
    total_pop: list[Schedule] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    @property
    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.total_pop if p.rank == 0]

    @property
    def avg_island_fit(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        return np.asarray([i.fitness_history.get_last_gen_fitness() for i in self.islands], dtype=float).mean(axis=0)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        try:
            self.notify_on_start()
            logger.debug("Seeding population...")
            self.seed_population()
            logger.debug("Initializing population...")
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                msg = "No valid schedule meeting all hard constraints was found."
                raise ValueError(msg)
            self.run_epochs()
        except KeyboardInterrupt:
            logger.warning("Genetic algorithm run interrupted by user. Saving...")
        finally:
            self._deduplicate_population()
            self.notify_on_finish()

    def seed_population(self) -> None:
        """Seed the population for each island."""
        if self.seed_pop:
            for ii, s_idx in self.island_seed_map.items():
                island = self.islands[ii]
                for si in s_idx:
                    if island.can_add(self.seed_pop[si]):
                        island.add(self.seed_pop[si])

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        for island in self.islands:
            island.initialize()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.gen_arr:
            for island in self.islands:
                island.run_epoch()
            self.fitness_history.current = self.avg_island_fit
            self.fitness_history.update_fitness_history()
            self.notify_on_generation_end(gen)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = [ind for island in self.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for s in unique_pop], dtype=int)
        schedule_fitness, team_fitnesses = self.ctx.evaluate(pop_array)
        _, flat, ranks = self.ctx.select_nsga3(schedule_fitness, len(unique_pop))
        selected = set()
        for rank, idx in zip(ranks, flat, strict=True):
            idx: int
            sch = unique_pop[idx]
            sch.fitness = schedule_fitness[idx]
            sch.team_fitnesses = team_fitnesses[idx]
            sch.rank = rank
            selected.add(sch)
        self.total_pop = sorted(selected, key=lambda s: (s.rank, -s.fitness.sum()))

    def notify_on_start(self) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.obs:
            obs.on_start(self.gen_arr.size)

    def notify_on_generation_end(self, gen: int) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.obs:
            obs.on_generation_end(gen, self.gen_arr.size, self.fitness_history.get_last_gen_fitness(), len(self))

    def notify_on_finish(self) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.obs:
            obs.on_finish(len(self.total_pop), len(self.pareto_front))


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    ctx: GaContext
    rng: np.random.Generator
    n_pop: int
    n_offspring: int
    chance_crossover: float
    chance_mutation: float
    opstat: OperatorStats
    fitness_history: FitnessHistory
    stagnation: StagnationHandler
    population: np.ndarray = field(default_factory=lambda: np.array([]))
    selected: list[Schedule] = field(default_factory=list)
    generation: int = 0
    _last_stagnant_gen: int = field(init=False, default=0)

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return self.population.shape[0] if self.population is not None else 0

    @property
    def n_needed(self) -> int:
        """Return the number of individuals needed to fill the population."""
        return self.n_pop - len(self)

    def add(self, s: Schedule) -> None:
        """Add a new schedule to the population."""
        self.selected.append(s)
        if self.population.size == 0:
            self.population = s.schedule[np.newaxis, :]
        else:
            self.population = np.vstack((self.population, s.schedule[np.newaxis, :]))

    def can_add(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        self.opstat.count_offspring("total")
        if self.ctx.check(schedule):
            self.opstat.count_offspring("success")
            if schedule not in self.selected:
                return True
        return False

    def _build_n_schedules(self, needed: int) -> None:
        """Build a number of schedules."""
        created = 0
        while created < needed:
            s = self.ctx.build()
            if self.ctx.repair(s) and self.can_add(s):
                self.add(s)
                created += 1

    def _handle_underpopulation(self) -> None:
        """Handle underpopulation by creating new individuals."""
        if self.n_needed <= 0:
            return
        logger.debug("Island %d: Handling underpopulation with %d individuals", self.identity, self.n_needed)
        self._build_n_schedules(self.n_needed)

    def _check_stagnation(self) -> None:
        """Check for stagnation without modifying the population."""
        if self.stagnation.is_stagnant(self.generation, self.fitness_history.history, self._last_stagnant_gen):
            self._last_stagnant_gen = self.generation
            idx_to_delete = self.rng.choice(np.arange(len(self.selected)))
            self.selected.pop(idx_to_delete)
            self.population = np.delete(arr=self.population, obj=idx_to_delete, axis=0)
            logger.debug(
                "Stagnation. Island: %d. Generation: %d. Schedule Removed: %d.",
                self.identity,
                self.generation + 1,
                idx_to_delete,
            )

    def _get_operator(self, ops: tuple) -> Any:
        """Randomly select an operator from a tuple of operators."""
        return ops[self.rng.integers(0, len(ops))]

    def _mutate_child(self, schedule: Schedule) -> bool:
        """Mutate a child schedule."""
        m: Mutation = self._get_operator(self.ctx.mutations)
        m_str = str(m)
        self.opstat.count_mutation("total", m_str)
        if m.mutate(schedule):
            self.opstat.count_mutation("success", m_str)
            schedule.mutations += 1
            return True
        return False

    def _crossover_parents(self, parents: Iterator[Schedule]) -> Iterator[Schedule]:
        """Perform crossover between two parent schedules."""
        c: Crossover = self._get_operator(self.ctx.crossovers)
        c_str = str(c)
        for child in c.cross(parents):
            self.opstat.count_crossover("total", c_str)
            if self.ctx.check(child):
                self.opstat.count_crossover("success", c_str)
        if self.ctx.repair(child):
            yield child

    def _evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (pop := self.selected):
            return
        created_cycle = 0
        while created_cycle < self.n_offspring:
            parents_indices = self.ctx.select_parents(n=len(pop))
            parents: Iterator[Schedule] = (pop[i] for i in parents_indices)
            c_roll = self.chance_crossover > self.rng.random()
            if c_roll and len(self.ctx.crossovers) > 0:
                offspring = self._crossover_parents(parents)
            else:
                offspring = (p.clone() for p in parents)
            for child in offspring:
                if created_cycle >= self.n_offspring:
                    break
                if len(self.ctx.mutations) > 0:
                    m_roll = not c_roll or self.chance_mutation > self.rng.random()
                    if m_roll:
                        self._mutate_child(child)
                if self.can_add(child):
                    self.add(child)
                created_cycle += 1
        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def _select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        schedule_fits, _ = self.ctx.evaluate(self.population)
        if schedule_fits.shape[0] != self.n_pop:
            _, flat, _ = self.ctx.select_nsga3(schedule_fits, self.n_pop)
        else:
            flat = np.arange(self.n_pop)
        self.fitness_history.current = schedule_fits[flat].mean(axis=0)
        total_pop: list[Schedule] = self.selected
        self.selected = []
        for i in flat:
            if self.can_add(total_pop[i]):
                self.selected.append(total_pop[i])
        self.population = self.population[flat]

    def run_epoch(self) -> None:
        """Run a full epoch: evaluate, select, evolve, and handle stagnation."""
        self._handle_underpopulation()
        self._evolve()
        self._select_next_generation()
        self.fitness_history.update_fitness_history()
        self._check_stagnation()
        self._handle_underpopulation()
        self.generation += 1

    def initialize(self) -> None:
        """Initialize the population for each island."""
        if self.n_needed <= 0:
            logger.debug("Island %d: Population already full with %d individuals", self.identity, len(self))
            return
        logger.debug("Island %d: Initializing population with %d individuals", self.identity, self.n_needed)
        self._build_n_schedules(self.n_needed)


@dataclass(slots=True)
class FitnessHistory:
    """Abstract base class for fitness history tracking."""

    generation: int
    current: np.ndarray
    history: np.ndarray

    def copy(self) -> FitnessHistory:
        """Create a copy of the fitness history."""
        return FitnessHistory(generation=self.generation, current=self.current.copy(), history=self.history.copy())

    def get_last_gen_fitness(self) -> np.ndarray:
        """Get the fitness of the last generation."""
        return (
            self.history[self.generation - 1] if self.generation > 0 else np.zeros(self.history.shape[1], dtype=float)
        )

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitnesses."""
        self.history[self.generation] = self.current
        self.generation += 1


@dataclass(slots=True)
class StagnationHandler:
    """Class for handling stagnation in the genetic algorithm."""

    enabled: bool
    threshold: int
    proportion: float
    cooldown: int

    def is_stagnant(self, curr: int, history: np.ndarray, last_stagnant_gen: int) -> bool:
        """Check if the GA has stagnated based on fitness history."""
        if not self.enabled or curr < self.threshold:
            return False
        # Get recent history
        recents = history[curr - self.threshold : curr]
        # Checks if any of the recent fitnesses exactly the same as the first in this range
        equal_mask = recents[0] == recents[1:]
        # Count how many are equal in all objectives
        equal_sum = equal_mask.sum(axis=1) > 0
        equal_count = equal_sum.sum()
        # Determine stagnation
        return equal_count > self.threshold * self.proportion and curr - last_stagnant_gen >= self.cooldown


@dataclass(slots=True)
class OperatorStats:
    """Class for collecting statistics on genetic operators."""

    offspring: Counter
    crossover: dict[str, Counter]
    mutation: dict[str, Counter]

    def count_offspring(self, op_status: str) -> None:
        """Record the use of an offspring operator (adding to population)."""
        self.offspring[op_status] += 1

    def count_crossover(self, op_status: str, op_name: str) -> None:
        """Record the use of a crossover operator."""
        self.crossover[op_status][op_name] += 1

    def count_mutation(self, op_status: str, op_name: str) -> None:
        """Record the use of a mutation operator."""
        self.mutation[op_status][op_name] += 1

    def get_offspring_stats(self) -> tuple[int, int, str]:
        """Get the offspring statistics."""
        s_sum = self.offspring.get("success", 0)
        t_sum = self.offspring.get("total", 0)
        return s_sum, t_sum, f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"

    def _get_operator_stats(self, operators: dict[str, Counter]) -> tuple[int, int, str]:
        """Get the statistics for a specific set of operators."""
        s_sum = sum(operators.get("success", Counter()).values())
        t_sum = sum(operators.get("total", Counter()).values())
        return s_sum, t_sum, f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"

    def get_crossover_stats(self) -> tuple[int, int, str]:
        """Get the crossover statistics for a specific operator."""
        return self._get_operator_stats(self.crossover)

    def get_mutation_stats(self) -> tuple[int, int, str]:
        """Get the mutation statistics for a specific operator."""
        return self._get_operator_stats(self.mutation)
