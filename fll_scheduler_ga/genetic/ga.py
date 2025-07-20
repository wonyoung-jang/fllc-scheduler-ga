"""Genetic algorithm for FLL Scheduler GA."""

import logging
import shelve
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from ..config.config import TournamentConfig
from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..observers.base_observer import GaObserver
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
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    ga_params: GaParameters
    config: TournamentConfig
    rng: Random
    event_factory: EventFactory
    team_factory: TeamFactory
    selections: tuple[Selection]
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
    logger: logging.Logger
    observers: tuple[GaObserver]
    evaluator: FitnessEvaluator
    repairer: Repairer
    builder: ScheduleBuilder = field(init=False, repr=False)
    nsga3: NSGA3 = field(default=None, init=False, repr=False)
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    population: Population = field(default_factory=list, init=False, repr=False)

    _population_hashes: set = field(default_factory=set, init=False, repr=False)
    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _mutation_ratio: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        self.builder = ScheduleBuilder(self.team_factory, self.event_factory, self.config, seeder)
        self.nsga3 = NSGA3(self.rng, len(self.evaluator.objectives), self.ga_params.population_size)

    def set_seed_file(self, file_path: str | Path | None) -> None:
        """Set the file path for loading a seed population."""
        if file_path:
            self._seed_file = Path(file_path)

    def pareto_front(self) -> Population:
        """Get the current Pareto front from the population."""
        if not self.population:
            return []
        return [p for p in self.population if p.rank == 0]

    def _calculate_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        front = self.pareto_front()
        if not front:
            return ()

        avg_fitness_front1 = defaultdict(list)

        for p in front:
            for i, _ in enumerate(self.evaluator.objectives):
                avg_fitness_front1[i].append(p.fitness[i])

        return tuple(sum(s) / len(s) for s in avg_fitness_front1.values())

    def _update_fitness_history(self) -> tuple[float, ...]:
        """Update the fitness history with the current generation's fitness."""
        if this_gen_fitness := self._calculate_this_gen_fitness():
            self.fitness_history.append(this_gen_fitness)
        return this_gen_fitness

    def _add_to_population(self, schedule: Schedule) -> bool:
        """Add a schedule to population if it's not a duplicate."""
        if hash(schedule) not in self._population_hashes:
            self.population.append(schedule)
            self._population_hashes.add(hash(schedule))
            return True
        return False

    def _cleanup_population_tracking(self) -> None:
        """Clean up population tracking structures after selection."""
        self._population_hashes = {hash(s) for s in self.population}

    def _load_population_from_seed(self) -> None:
        """Load and integrate a population from a seed file."""
        self.logger.info("Loading seed population from: %s", self._seed_file)
        try:
            with shelve.open(self._seed_file) as shelf:
                shelf_config: TournamentConfig = shelf.get("config")
                if (
                    self.config.num_teams != shelf_config.num_teams
                    or self.config.rounds != shelf_config.rounds
                    or self.config.round_requirements != shelf_config.round_requirements
                    or self.config.total_slots != shelf_config.total_slots
                ):
                    self.logger.warning(
                        "Seed file configuration does not match current configuration. Using current config."
                    )
                    return
                seeded_population = shelf["population"]

            added_count = 0
            for schedule in seeded_population:
                if self._add_to_population(schedule):
                    added_count += 1

            self.logger.info("Loaded %d unique individuals from seed file.", added_count)
        except (OSError, KeyError, EOFError, Exception):
            self.logger.exception("Could not load or parse seed file. Starting with a fresh population.")
            self.population.clear()
            self._population_hashes.clear()

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time.time()
        self._notify_on_start(self.ga_params.generations)
        try:
            self.initialize_population()
            if not self.population:
                self.logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            if self.generation():
                self.population = self.nsga3.select(self.population)
                self._notify_on_finish(self.population, self.pareto_front())
                return True
        except Exception:
            self.logger.exception("An error occurred during the genetic algorithm run.")
            return False
        except KeyboardInterrupt:
            self.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            for p in self.population:
                if p.fitness is None:
                    p.fitness = self.evaluator.evaluate(p)
            self.population = self.nsga3.select(self.population)
            self._update_fitness_history()
            self._notify_on_finish(self.population, self.pareto_front())
            return True
        finally:
            if start_time:
                total_time = time.time() - start_time
                self.logger.info("Total time taken: %.2f seconds", total_time)
        return True

    def initialize_population(self) -> None:
        """Initialize the population with random organisms."""
        if self._seed_file and self._seed_file.exists():
            self._load_population_from_seed()

        num_to_create = self.ga_params.population_size - len(self.population)
        if num_to_create <= 0:
            self._cleanup_population_tracking()
            self._update_fitness_history()
            return

        self.logger.info("Initializing population with %d individuals.", num_to_create)
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        attempts, max_attempts = ATTEMPTS

        num_created = 0
        while len(self.population) < num_to_create and attempts < max_attempts:
            self.builder.rng = Random(seeder.randint(*RANDOM_SEED))
            schedule = self.builder.build()
            if self.repairer.repair(schedule) and self._add_to_population(schedule):
                schedule.fitness = self.evaluator.evaluate(schedule)
                num_created += 1
            else:
                attempts += 1
                self.logger.info("Attempt %d: Created %d valid schedules so far.", attempts, num_created)

        if not self.population or not self.population[0].fitness:
            self.logger.critical("No valid initial schedules could be built.")
            return

        self.population = self.nsga3.select(self.population)
        self._cleanup_population_tracking()
        self._update_fitness_history()

    def initialize_schedules(self, num_schedules: int) -> None:
        """Initialize a set of schedules."""
        schedule_count = 0
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        while schedule_count < num_schedules:
            self.builder.rng = Random(seeder.randint(*RANDOM_SEED))
            schedule = self.builder.build()
            if self.repairer.repair(schedule) and self._add_to_population(schedule):
                schedule.fitness = self.evaluator.evaluate(schedule)
                schedule_count += 1

    def generation(self) -> bool:
        """Perform a single generation step of the genetic algorithm."""
        num_offspring = self.ga_params.population_size - self.ga_params.elite_size
        for generation in range(self.ga_params.generations):
            self.evolve(num_offspring)
            self.population = self.nsga3.select(self.population)
            if len(self.population) < self.ga_params.population_size:
                self.initialize_schedules(self.ga_params.population_size - len(self.population))
            self._cleanup_population_tracking()
            self._notify_gen_end(generation, self._update_fitness_history())
            if not self.population:
                self.logger.warning("No valid individuals in the current population.")
                return False

        return True

    def evolve(self, num_offspring: int) -> None:
        """Evolve the population to create a new generation."""
        child_count = 0
        while child_count < num_offspring:
            s = self.rng.choice(self.selections)
            parents = tuple(s.select(self.population, num_parents=2))

            for child in self.crossover_child(parents):
                self.mutate_child(child)
                if self.repairer.repair(child) and self._add_to_population(child):
                    child.fitness = self.evaluator.evaluate(child)
                    child_count += 1
                    self._offspring_ratio["success"] += 1
                else:
                    self._offspring_ratio["failure"] += 1

                if child_count >= num_offspring:
                    break

    def crossover_child(self, parents: tuple[Schedule, ...]) -> Iterator[Schedule]:
        """Evolve the population by one individual and return the best of the parents and child."""
        if self.ga_params.crossover_chance > self.rng.random():
            yield from self.rng.choice(self.crossovers).crossover(parents)

    def mutate_child(self, child: Schedule) -> None:
        """Mutate the child schedule."""
        if self.ga_params.mutation_chance > self.rng.random():
            mutation_success = self.rng.choice(self.mutations).mutate(child)
            self._mutation_ratio["success" if mutation_success else "failure"] += 1

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_gen_end(self, generation: int, best_fitness: tuple[float, ...]) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(
                generation + 1,
                self.ga_params.generations,
                len(self.population),
                best_fitness,
                len(self.pareto_front()),
            )

    def _notify_on_finish(self, pop: Population, front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        o_success = self._offspring_ratio["success"]
        o_total = o_success + self._offspring_ratio["failure"]
        o_success_percentage = o_success / o_total if o_total > 0 else 0.0

        m_success = self._mutation_ratio["success"]
        m_total = m_success + self._mutation_ratio["failure"]
        m_success_percentage = m_success / m_total if m_total > 0 else 0.0

        self.logger.info(
            "Offspring success ratio: %s/%s = %s | %s",
            o_success,
            o_total,
            f"{o_success_percentage:.2%}",
            self._offspring_ratio,
        )
        self.logger.info(
            "Mutation success ratio: %s/%s = %s | %s",
            m_success,
            m_total,
            f"{m_success_percentage:.2%}",
            self._mutation_ratio,
        )
        self.logger.info(
            "Unique/Total individuals: %s/%s",
            len({hash(s) for s in self.population}),
            len(self.population),
        )
        for obs in self.observers:
            obs.on_finish(pop, front)
