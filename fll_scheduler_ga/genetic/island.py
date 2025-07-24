"""Genetic algorithm for FLL Scheduler GA."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from random import Random

from ..config.config import TournamentConfig
from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import Selection
from .builder import ScheduleBuilder
from .fitness import FitnessEvaluator
from .ga_parameters import GaParameters
from .schedule import Population, Schedule

RANDOM_SEED = (1, 2**32 - 1)
ATTEMPTS = (0, 50)


@dataclass(slots=True)
class Island:
    """Genetic algorithm island for the FLL Scheduler GA."""

    identity: int
    ga_params: GaParameters
    config: TournamentConfig
    rng: Random
    event_factory: EventFactory
    team_factory: TeamFactory
    selections: tuple[Selection]
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
    logger: logging.Logger
    evaluator: FitnessEvaluator
    repairer: Repairer

    builder: ScheduleBuilder = field(init=False, repr=False)
    nsga3: NSGA3 = field(default=None, init=False, repr=False)

    population: Population = field(default_factory=list, init=False, repr=False)
    hashes: list[int] = field(default_factory=list, init=False, repr=False)

    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _mutation_ratio: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        _randlow, _randhigh = RANDOM_SEED
        seeder = Random(self.rng.randint(_randlow, _randhigh))
        self.builder = ScheduleBuilder(
            self.team_factory,
            self.event_factory,
            self.config,
            Random(seeder.randint(_randlow, _randhigh)),
        )
        self.repairer.rng = Random(seeder.randint(_randlow, _randhigh))
        self.nsga3 = NSGA3(self.rng, len(self.evaluator.objectives), self.ga_params.population_size)

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.population if p.rank == 0]

    def add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        schedule_hash = hash(schedule)
        if schedule_hash not in self.hashes:
            self.population.append(schedule)
            self.hashes.append(schedule_hash)
            return True
        return False

    def initialize(self) -> None:
        """Initialize the population for each island."""
        num_to_create = self.ga_params.population_size - len(self.hashes)
        self.logger.info("Initializing island %d with %d individuals.", self.identity, num_to_create)
        if num_to_create <= 0:
            return

        _randlow, _randhigh = RANDOM_SEED
        seeder = Random(self.rng.randint(_randlow, _randhigh))
        attempts, max_attempts = ATTEMPTS
        num_created = 0

        while len(self.hashes) < self.ga_params.population_size and attempts < max_attempts:
            self.builder.rng = Random(seeder.randint(_randlow, _randhigh))
            schedule = self.builder.build()

            if self.repairer.repair(schedule) and self.add_to_population(schedule):
                schedule.fitness = self.evaluator.evaluate(schedule)
                num_created += 1
            elif num_created == 0:
                # Only increment attempts if no valid schedule was created in total
                attempts += 1

        if num_created < num_to_create:
            self.logger.warning(
                "Island %d: only created %d/%d valid individuals.",
                self.identity,
                num_created,
                num_to_create,
            )

        self.population = self.nsga3.select(self.population)
        self.hashes = [hash(s) for s in self.population]

    def evolve(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        island_pop = self.population
        if not island_pop:
            return None

        num_offspring = self.ga_params.population_size - self.ga_params.elite_size
        attempts, max_attempts = 0, num_offspring * 5
        child_count = 0

        repair = self.repairer.repair
        evaluate = self.evaluator.evaluate
        offspring_ratio = self._offspring_ratio
        mutation_ratio = self._mutation_ratio
        choose = self.rng.choice
        roll = self.rng.random
        crossover_chance = self.ga_params.crossover_chance
        mutation_chance = self.ga_params.mutation_chance

        while child_count < num_offspring and attempts < max_attempts:
            attempts += 1
            parents = tuple(choose(self.selections).select(island_pop, num_parents=2))
            if len(set(parents)) < 2:
                continue

            if crossover_chance < roll():
                continue

            for child in choose(self.crossovers).crossover(parents):
                if mutation_chance > roll():
                    mutation_success = choose(self.mutations).mutate(child)
                    mutation_ratio["success" if mutation_success else "failure"] += 1

                if repair(child) and self.add_to_population(child):
                    child.fitness = evaluate(child)
                    child_count += 1
                    offspring_ratio["success"] += 1
                else:
                    offspring_ratio["failure"] += 1

                if child_count >= num_offspring:
                    break

        self.population = self.nsga3.select(self.population)
        self.hashes = [hash(s) for s in self.population]

        return {
            "offspring": offspring_ratio,
            "mutation": mutation_ratio,
        }

    def get_migrants(self) -> list[Schedule]:
        """Get the list of migrants from the current island."""
        migration_size = self.ga_params.migration_size
        if migration_size == 0:
            return []

        pareto_front = self.pareto_front()
        return self.rng.sample(pareto_front, k=min(migration_size, len(pareto_front)))

    def receive_migrants(self, migrants: list[Schedule]) -> None:
        """Receive migrants from another island and add them to the current island's population."""
        for migrant in migrants:
            self.add_to_population(migrant)

        self.population = self.nsga3.select(self.population)
        self.hashes = [hash(s) for s in self.population]
