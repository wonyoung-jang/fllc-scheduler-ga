"""Genetic algorithm for FLL Scheduler GA."""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fll_scheduler_ga.adapter.monitoring import GaObserver
    from fll_scheduler_ga.adapter.schema import GaParameterModel, GeneticModel
    from fll_scheduler_ga.domain.model import Schedule
    from fll_scheduler_ga.genetic.context import GaContext
    from fll_scheduler_ga.genetic.operator import Crossover, Mutation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    genetic_model: GeneticModel
    rng: np.random.Generator
    observers: tuple[GaObserver, ...]
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    generations_array: np.ndarray
    migrate_generations: np.ndarray
    seed_pop: list[Schedule] = field(default_factory=list)
    island_seed_map: dict[int, list[int]] = field(default_factory=dict)
    total_population: list[Schedule] = field(default_factory=list)
    islands: list[Island] = field(default_factory=list)
    start_time: float = 0.0

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        for i in range(self.genetic_model.parameters.num_islands):
            self.islands.append(
                Island(
                    identity=i,
                    context=self.context,
                    ga_param=self.genetic_model.parameters,
                    rng=self.rng,
                    operator_stats=self.operator_stats,
                    fitness_history=self.fitness_history.copy(),
                    stagnation=StagnationHandler(
                        enabled=self.genetic_model.stagnation.enable,
                        threshold=self.genetic_model.stagnation.threshold,
                        proportion=self.genetic_model.stagnation.proportion,
                        cooldown=self.genetic_model.stagnation.cooldown,
                    ),
                )
            )

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    @property
    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.total_population if p.rank == 0]

    @property
    def avg_island_fit(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        return np.asarray([i.fitness_history.get_last_gen_fitness() for i in self.islands], dtype=float).mean(axis=0)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        try:
            self.start_time = time.perf_counter()
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
            logger.exception("Genetic algorithm run interrupted by user. Saving...")
        finally:
            self._deduplicate_population()
            self.notify_on_finish()

    def seed_population(self) -> None:
        """Seed the population for each island."""
        if self.seed_pop:
            for ii, s_idx in self.island_seed_map.items():
                island = self.islands[ii]
                for si in s_idx:
                    if island.add_to_population(self.seed_pop[si]):
                        island.add(self.seed_pop[si].schedule)

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        for island in self.islands:
            island.initialize()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.generations_array:
            if self.migrate_generations[gen]:
                self.migrate()
            # Run the generations
            for island in self.islands:
                island.run_epoch()
            self.fitness_history.current = self.avg_island_fit
            self.fitness_history.update_fitness_history()
            self.notify_on_generation_end(gen)

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        n = self.genetic_model.parameters.num_islands
        for i, dest in enumerate(self.islands):
            src = self.islands[(i + 1) % n]
            migrants = src.give_migrants()
            dest.receive_migrants(migrants)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = [ind for island in self.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in self.islands for s in island.selected])
        schedule_fitness, team_fitnesses = self.context.evaluate(pop_array)
        _, flat, ranks = self.context.select_nsga3(schedule_fitness, len(unique_pop))
        selected = set()
        for rank, idx in zip(ranks, flat, strict=True):
            idx: int
            sch = unique_pop[idx]
            sch.fitness = schedule_fitness[idx]
            sch.team_fitnesses = team_fitnesses[idx]
            sch.rank = rank
            selected.add(sch)
        self.total_population = sorted(selected, key=lambda s: (s.rank, -s.fitness.sum()))

    def notify_on_start(self) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(self.genetic_model.parameters.generations)

    def notify_on_generation_end(self, gen: int) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(
                gen, self.genetic_model.parameters.generations, self.fitness_history.get_last_gen_fitness(), len(self)
            )

    def notify_on_finish(self) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(len(self.total_population), len(self.pareto_front))


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
    population: np.ndarray = field(default_factory=lambda: np.array([]))
    selected: list[Schedule] = field(default_factory=list)
    generation: int = 0
    curr_schedule_fits: np.ndarray = field(init=False)

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return self.population.shape[0] if self.population is not None else 0

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
            self.population = np.delete(arr=self.population, obj=idx_to_pop, axis=0)
            logger.debug(
                "Stagnation. Island: %d. Generation: %d. Schedule Removed: %d.",
                self.identity,
                self.generation + 1,
                idx_to_pop,
            )

    def add(self, schedule: np.ndarray) -> None:
        """Add a new schedule to the population."""
        if self.population.size == 0:
            self.population = schedule[np.newaxis, :]
        else:
            self.population = np.vstack((self.population, schedule[np.newaxis, :]))

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
                self.add(s.schedule)
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
                    m_roll = not c_roll or self.ga_param.mutation_chance > self.rng.random()
                    if m_roll:
                        self._mutate_child(child)
                if self.add_to_population(child):
                    self.add(child.schedule)
                created_cycle += 1
        if not self.selected:
            msg = f"Island {self.identity}: No individuals in population after evolution."
            raise RuntimeError(msg)

    def _select_next_generation(self) -> None:
        """Select the next generation using NSGA-III principles."""
        n_pop = self.ga_param.population_size
        schedule_fits, _ = self.context.evaluate(self.population)
        if schedule_fits.shape[0] != n_pop:
            _, flat, _ = self.context.select_nsga3(schedule_fits, n_pop)
        else:
            flat = np.arange(n_pop)
        self.fitness_history.current = schedule_fits[flat].mean(axis=0)
        total_pop: list[Schedule] = self.selected
        self.selected = []
        for i in flat:
            self.add_to_population(total_pop[i])
        self.population = self.population[flat]
        self.curr_schedule_fits = schedule_fits[flat]

    def give_migrants(self) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        for _ in range(self.ga_param.migration_size):
            i = self.rng.integers(low=0, high=len(self.selected))
            self.population = np.delete(self.population, i, axis=0)
            yield self.selected.pop(i)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            if self.add_to_population(schedule=migrant):
                self.add(migrant.schedule)


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
    _last_stagnant_gen: int = 0

    def is_stagnant(self, curr: int, history: np.ndarray) -> bool:
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
        if equal_count > self.threshold * self.proportion and curr - self._last_stagnant_gen >= self.cooldown:
            self._last_stagnant_gen = curr
            return True
        return False


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
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate

    def _get_operator_stats(self, operators: dict[str, Counter]) -> tuple[int, int, str]:
        """Get the statistics for a specific set of operators."""
        s_sum = sum(operators.get("success", Counter()).values())
        t_sum = sum(operators.get("total", Counter()).values())
        rate = f"{s_sum / t_sum if t_sum > 0 else 0.0:.2%}"
        return s_sum, t_sum, rate

    def get_crossover_stats(self) -> tuple[int, int, str]:
        """Get the crossover statistics for a specific operator."""
        return self._get_operator_stats(self.crossover)

    def get_mutation_stats(self) -> tuple[int, int, str]:
        """Get the mutation statistics for a specific operator."""
        return self._get_operator_stats(self.mutation)
