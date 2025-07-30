"""Genetic algorithm for FLL Scheduler GA."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from random import Random

from ..config.constants import ATTEMPTS_RANGE, RANDOM_SEED_RANGE
from .builder import ScheduleBuilder
from .ga_context import GaContext
from .schedule import Population, Schedule


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    rng: Random
    builder: ScheduleBuilder
    context: GaContext

    population: Population = field(default_factory=list, init=False, repr=False)
    hashes: set[int] = field(default_factory=set, init=False, repr=False)

    def __len__(self) -> int:
        """Return the number of individuals in the island's population."""
        return len(self.population)

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.population if p.rank == 0]

    def add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        schedule_hash = hash(schedule)
        if schedule_hash not in self.hashes:
            self.population.append(schedule)
            self.hashes.add(schedule_hash)
            return True
        return False

    def initialize(self) -> None:
        """Initialize the population for each island."""
        pop_size = self.context.ga_params.population_size
        num_to_create = pop_size - len(self.population)
        if num_to_create <= 0:
            self.context.logger.info("Initializing island %d with 0 individuals.", self.identity)
            return
        self.context.logger.info("Initializing island %d with %d individuals.", self.identity, num_to_create)

        _randlow, _randhigh = RANDOM_SEED_RANGE
        seeder = Random(self.rng.randint(_randlow, _randhigh))
        attempts, max_attempts = ATTEMPTS_RANGE
        num_created = 0

        while len(self.population) < pop_size and attempts < max_attempts:
            self.builder.rng = Random(seeder.randint(_randlow, _randhigh))
            schedule = self.builder.build()

            if self.context.repairer.repair(schedule) and self.add_to_population(schedule):
                schedule.fitness = self.context.evaluator.evaluate(schedule)
                num_created += 1
            elif num_created == 0:
                # Only increment attempts if no valid schedule was created in total
                attempts += 1

        if num_created < num_to_create:
            self.context.logger.warning(
                "Island %d: only created %d/%d valid individuals.",
                self.identity,
                num_created,
                num_to_create,
            )

        self.population, self.hashes = self.context.nsga3.select(self.population, population_size=pop_size)

    def evolve(self) -> dict[str, Counter]:
        """Perform main evolution loop: generations and migrations."""
        offspring_ratio = Counter()
        mutation_ratio = Counter()

        if not (island_pop := self.population):
            return {
                "offspring": offspring_ratio,
                "mutation": mutation_ratio,
            }

        _context = self.context
        _ga_params = _context.ga_params
        num_offspring = _ga_params.population_size - _ga_params.elite_size
        attempts, max_attempts = 0, num_offspring * 5
        child_count = 0

        while child_count < num_offspring and attempts < max_attempts:
            attempts += 1
            selection_op = self.rng.choice(_context.selections)
            parents = tuple(selection_op.select(island_pop, num_parents=2))
            if parents[0] == parents[1]:
                continue

            if _ga_params.crossover_chance <= self.rng.random():
                continue

            crossover_op = self.rng.choice(_context.crossovers)
            for child in crossover_op.crossover(parents):
                mutation_ops = self.rng.sample(_context.mutations, k=len(_context.mutations))
                for i, mutation_op in enumerate(mutation_ops, start=1):
                    mutation_chance = _ga_params.mutation_chance**i
                    if mutation_chance <= self.rng.random():
                        continue

                    mutation_success = mutation_op.mutate(child)
                    mutation_ratio["success" if mutation_success else "failure"] += 1

                if _context.repairer.repair(child) and self.add_to_population(child):
                    child.fitness = _context.evaluator.evaluate(child)
                    child_count += 1
                    offspring_ratio["success"] += 1
                else:
                    offspring_ratio["failure"] += 1

                if child_count >= num_offspring:
                    break

        self.population, self.hashes = _context.nsga3.select(
            self.population, population_size=_ga_params.population_size
        )

        return {
            "offspring": offspring_ratio,
            "mutation": mutation_ratio,
        }

    def get_migrants(self, migration_size: int) -> Iterator[Schedule]:
        """Randomly yield migrants from population."""
        yield from self.rng.sample(self.population, k=migration_size)

    def receive_migrants(self, migrants: Iterator[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)

        self.population, self.hashes = self.context.nsga3.select(
            self.population, population_size=self.context.ga_params.population_size
        )

    def finalize_island(self) -> Iterator[Schedule]:
        """Finalize the island's state after evolution."""
        evaluate = self.context.evaluator.evaluate
        for sched in self.population:
            if sched.fitness is None:
                sched.fitness = evaluate(sched)
            yield sched
