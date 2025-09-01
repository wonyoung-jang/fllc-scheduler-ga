"""Genetic algorithm for FLL Scheduler GA."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from logging import getLogger
from random import Random

from ..config.constants import ATTEMPTS_RANGE, RANDOM_SEED_RANGE
from ..config.ga_context import GaContext
from ..config.ga_parameters import GaParameters
from ..data_model.schedule import Population, Schedule
from .builder import ScheduleBuilder

logger = getLogger(__name__)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    rng: Random
    builder: ScheduleBuilder
    context: GaContext
    ga_params: GaParameters

    selected: dict[int, Schedule] = field(default_factory=dict, init=False, repr=False)
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    crossover_ratio: dict = field(default=None, init=False, repr=False)
    mutation_ratio: dict = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.crossover_ratio = {tracker: Counter() for tracker in ("success", "total")}
        self.mutation_ratio = {tracker: Counter() for tracker in ("success", "total")}

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def _get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        # if not (pop := self.pareto_front()):
        if not (pop := self.selected.values()):
            return ()

        len_objectives = len(self.context.evaluator.objectives)
        avg_fits = [0 for _ in range(len_objectives)]
        for p in pop:
            for i in range(len_objectives):
                avg_fits[i] += p.fitness[i]

        return tuple(avg_fit / len(pop) for avg_fit in avg_fits)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        this_gen_fitness = self._get_this_gen_fitness()
        self.fitness_history.append(this_gen_fitness)

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.selected.values() if p.rank == 0]

    def add_to_population(self, schedule: Schedule, s_hash: int | None = None) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        schedule_hash = s_hash if s_hash else hash(schedule)
        if schedule_hash in self.selected:
            return False
        self.selected[schedule_hash] = schedule
        return True

    def _nsga3_select(self) -> None:
        """NSGA-III selection."""
        population = list(self.selected.values())
        self.selected = self.context.nsga3.select(
            population=population,
            population_size=self.ga_params.population_size,
        )
        self.handle_underpopulation()

    def initialize(self) -> None:
        """Initialize the population for each island."""
        pop_size = self.ga_params.population_size
        num_to_create = pop_size - len(self.selected)
        if num_to_create <= 0:
            logger.debug("Initializing island %d with 0 individuals.", self.identity)
            return

        logger.debug("Initializing island %d with %d individuals.", self.identity, num_to_create)

        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        attempts, max_attempts = ATTEMPTS_RANGE
        num_created = 0

        while len(self.selected) < pop_size and attempts < max_attempts:
            schedule = self.builder.build(rng=Random(seeder.randint(*RANDOM_SEED_RANGE)))
            if self.context.repairer.repair(schedule) and self.add_to_population(schedule):
                self.context.evaluator.evaluate(schedule)
                num_created += 1
            elif num_created == 0:
                attempts += 1

        if num_created == 0:
            msg = "Island %d: No valid individuals created after %d attempts. Try adjusting parameters."
            raise RuntimeError(msg % (self.identity, attempts))

        if num_created < num_to_create:
            logger.warning(
                "Island %d: only created %d/%d valid individuals.",
                self.identity,
                num_created,
                num_to_create,
            )

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (island_pop := list(self.selected.values())):
            return

        attempts, max_attempts = ATTEMPTS_RANGE
        child_count = 0

        while child_count < self.ga_params.offspring_size and attempts < max_attempts:
            attempts += 1
            _crossover_roll = self.ga_params.crossover_chance > self.rng.random()
            _mutation_roll = self.ga_params.mutation_chance > self.rng.random()

            selection_op = self.rng.choice(self.context.selections)
            parents: tuple[Schedule] = tuple(selection_op.select(island_pop))
            offspring: set[Schedule] = set()
            must_mutate = False

            if _crossover_roll and self.context.crossovers:
                crossover_op = self.rng.choice(self.context.crossovers)
                for child in crossover_op.crossover(parents):
                    _c_str = str(crossover_op)
                    self.crossover_ratio["total"][_c_str] += 1
                    if self.context.repairer.repair(child):
                        self.context.evaluator.evaluate(child)
                        offspring.add(child)
                        self.crossover_ratio["success"][_c_str] += 1
            else:
                must_mutate = True  # If no crossover is performed, we must mutate clones of parents
                offspring.update(p.clone() for p in parents)

            for child in offspring:
                if (must_mutate or _mutation_roll) and self.context.mutations:
                    mutation_op = self.rng.choice(self.context.mutations)
                    _m_str = str(mutation_op)
                    self.mutation_ratio["total"][_m_str] += 1
                    if not mutation_op.mutate(child):
                        continue

                    self.mutation_ratio["success"][_m_str] += 1
                    self.offspring_ratio["total"] += 1
                    if not (self.context.repairer.repair(child) and self.add_to_population(child)):
                        continue

                    self.context.evaluator.evaluate(child)
                    child_count += 1
                    self.offspring_ratio["success"] += 1

                if child_count >= self.ga_params.offspring_size:
                    break

        self._nsga3_select()

    def get_migrants(self) -> tuple[Schedule]:
        """Randomly yield migrants from population."""
        _keys = list(self.selected.keys())
        _shuffled_keys = self.rng.sample(_keys, k=self.ga_params.migration_size)
        return tuple(self.selected.pop(s_hash) for s_hash in _shuffled_keys)

    def receive_migrants(self, migrants: tuple[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)
        self._nsga3_select()

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by adding new individuals to the island."""
        curr_size = len(self.selected)
        if curr_size >= self.ga_params.population_size:
            return

        while curr_size < self.ga_params.population_size:
            choice: Schedule = None
            if self.rng.choice((True, False)):
                curr_pop = list(self.selected.values())
                choice_idx = self.rng.randint(0, len(curr_pop) - 1)
                choice = curr_pop[choice_idx].clone()
                for m in self.context.mutations:
                    _m_str = str(m)
                    self.mutation_ratio["total"][_m_str] += 1
                    if m.mutate(choice):
                        self.mutation_ratio["success"][_m_str] += 1
            else:
                seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
                choice = self.builder.build(rng=Random(seeder.randint(*RANDOM_SEED_RANGE)))

            if self.context.repairer.repair(choice) and self.add_to_population(choice):
                self.context.evaluator.evaluate(choice)
                curr_size += 1

    def finalize_island(self) -> Iterator[Schedule]:
        """Finalize the island's state after evolution."""
        for sched in self.selected.values():
            if sched.fitness is None:
                self.context.evaluator.evaluate(sched)
            yield sched
