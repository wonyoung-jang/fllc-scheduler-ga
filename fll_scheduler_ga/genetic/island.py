"""Genetic algorithm for FLL Scheduler GA."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from random import Random

from ..config.constants import ATTEMPTS_RANGE, RANDOM_SEED_RANGE
from ..data_model.schedule import Population, Schedule
from ..operators.crossover import Crossover
from .builder import ScheduleBuilder
from .ga_context import GaContext


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    rng: Random
    builder: ScheduleBuilder
    context: GaContext

    selected: dict[int, Schedule] = field(default_factory=dict, init=False, repr=False)
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    crossover_ratio: dict = field(default_factory=dict, init=False, repr=False)
    mutation_ratio: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.crossover_ratio = {tracker: Counter() for tracker in ("success", "total")}
        self.mutation_ratio = {tracker: Counter() for tracker in ("success", "total")}
        self.offspring_ratio = Counter()

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.selected)

    def _get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        if not (pop := self.pareto_front()):
            return ()

        len_objectives = len(self.context.evaluator.objectives)
        avg_fits = [0 for _ in range(len_objectives)]
        for p in pop:
            for i in range(len_objectives):
                avg_fits[i] += p.fitness[i]

        return tuple(s / len(pop) for s in avg_fits)

    def get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history.append(self._get_this_gen_fitness())

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.selected.values() if p.rank == 0]

    def add_to_population(self, schedule: Schedule, s_hash: int | None = None) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        schedule_hash = hash(schedule) if s_hash is None else s_hash
        if schedule_hash not in self.selected:
            self.selected[schedule_hash] = schedule
            return True
        return False

    def initialize(self) -> None:
        """Initialize the population for each island."""
        pop_size = self.context.ga_params.population_size
        num_to_create = pop_size - len(self.selected)
        if num_to_create <= 0:
            self.context.logger.info("Initializing island %d with 0 individuals.", self.identity)
            return
        self.context.logger.info("Initializing island %d with %d individuals.", self.identity, num_to_create)

        _randlow, _randhigh = RANDOM_SEED_RANGE
        seeder = Random(self.rng.randint(_randlow, _randhigh))
        attempts, max_attempts = ATTEMPTS_RANGE
        num_created = 0

        while len(self.selected) < pop_size and attempts < max_attempts:
            self.builder.rng = Random(seeder.randint(_randlow, _randhigh))
            schedule = self.builder.build()

            if self.context.repairer.repair(schedule) and self.add_to_population(schedule):
                schedule.fitness = self.context.evaluator.evaluate(schedule)
                num_created += 1
            elif num_created == 0:
                # Only increment attempts if no valid schedule was created in total
                attempts += 1

        if num_created == 0:
            msg = "Island %d: No valid individuals created after %d attempts. Try adjusting parameters."
            raise RuntimeError(msg % (self.identity, attempts))

        if num_created < num_to_create:
            self.context.logger.warning(
                "Island %d: only created %d/%d valid individuals.",
                self.identity,
                num_created,
                num_to_create,
            )

        self.selected = self.context.nsga3.select(self.selected.values(), population_size=pop_size)

    def get_initial_offspring(
        self, parents: tuple[Schedule, Schedule], cs: tuple[Crossover] | tuple, *, crossover_roll: bool
    ) -> set[Schedule]:
        """Get the initial offspring for the island from either crossover or parents."""
        offspring = set()
        if crossover_roll and cs:
            crossover_op = self.rng.choice(self.context.crossovers)
            for child in crossover_op.crossover(parents):
                self.crossover_ratio["total"][f"{crossover_op!s}"] += 1
                if self.context.repairer.repair(child):
                    child.fitness = self.context.evaluator.evaluate(child)
                    offspring.add(child)
                    self.crossover_ratio["success"][f"{crossover_op!s}"] += 1
        elif not cs:
            offspring.update(parents)
        return offspring

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        if not (island_pop := list(self.selected.values())):
            return

        _context = self.context
        _ss = _context.selections
        _cs = _context.crossovers
        _ms = _context.mutations
        _eval = _context.evaluator.evaluate
        _ga_params = _context.ga_params
        attempts, max_attempts = ATTEMPTS_RANGE
        child_count = 0

        while child_count < _ga_params.offspring_size and attempts < max_attempts:
            attempts += 1
            _crossover_roll = _ga_params.crossover_chance > self.rng.random()
            _mutation_roll = _ga_params.mutation_chance > self.rng.random()

            selection_op = self.rng.choice(_ss)
            parents = tuple(selection_op.select(island_pop, num_parents=2))

            offspring = self.get_initial_offspring(
                parents,
                _cs,
                crossover_roll=_crossover_roll,
            )

            for child in offspring:
                if _mutation_roll and _ms:
                    mutation_op = self.rng.choice(_ms)
                    self.mutation_ratio["total"][f"{mutation_op!s}"] += 1
                    if mutation_op.mutate(child):
                        self.mutation_ratio["success"][f"{mutation_op!s}"] += 1

                self.offspring_ratio["total"] += 1
                if self.add_to_population(child):
                    child.fitness = _eval(child)
                    child_count += 1
                    self.offspring_ratio["success"] += 1

                if child_count >= _ga_params.offspring_size:
                    break

        self.selected = self.context.nsga3.select(self.selected.values(), population_size=_ga_params.population_size)

    def get_migrants(self, migration_size: int) -> Iterator[tuple[int, Schedule]]:
        """Randomly yield migrants from population."""
        for migrant_hash in self.rng.sample(list(self.selected.keys()), k=migration_size):
            migrant = self.selected.pop(migrant_hash)
            yield (migrant_hash, migrant)

    def receive_migrants(self, migrants: Iterator[tuple[int, Schedule]]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant_hash, migrant in migrants:
            self.add_to_population(migrant, migrant_hash)

        self.selected = self.context.nsga3.select(
            self.selected.values(), population_size=self.context.ga_params.population_size
        )

    def handle_underpopulation(self) -> None:
        """Handle underpopulation by adding new individuals to the island."""
        if len(self.selected) >= self.context.ga_params.population_size:
            return

        self.context.logger.debug(
            "Island %d underpopulated: %d individuals, expected %d.",
            self.identity,
            len(self.selected),
            self.context.ga_params.population_size,
        )

        while len(self.selected) < self.context.ga_params.population_size:
            seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
            self.builder.rng = Random(seeder.randint(*RANDOM_SEED_RANGE))
            schedule = self.builder.build()

            self.offspring_ratio["total"] += 1
            if self.context.repairer.repair(schedule) and self.add_to_population(schedule):
                self.offspring_ratio["success"] += 1
                schedule.fitness = self.context.evaluator.evaluate(schedule)

    def finalize_island(self) -> Iterator[Schedule]:
        """Finalize the island's state after evolution."""
        for sched in self.selected.values():
            if sched.fitness is None:
                sched.fitness = self.context.evaluator.evaluate(sched)
            yield sched
