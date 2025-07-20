"""Genetic algorithm for FLL Scheduler GA."""

import logging
import pickle
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
from ..operators.nsga2 import NSGA2
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
    elitism: Selection
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
    logger: logging.Logger
    observers: tuple[GaObserver]
    evaluator: FitnessEvaluator
    repairer: Repairer
    nsga2: NSGA2 = field(default=None, init=False, repr=False)
    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    population: Population = field(default_factory=list, init=False, repr=False)

    _population_hashes: set = field(default_factory=set, init=False, repr=False)
    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _crossover_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _mutation_ratio: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.nsga2 = NSGA2

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

        num_objectives = len(front[0].fitness)
        avg_fitness_front1 = defaultdict(list)

        for p in front:
            for i in range(num_objectives):
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
        current_hashes = {hash(s) for s in self.population}
        self._population_hashes = current_hashes

    def _load_population_from_seed(self) -> None:
        """Load and integrate a population from a seed file."""
        self.logger.info("Loading seed population from: %s", self._seed_file)
        try:
            with self._seed_file.open("rb") as f:
                seeded_population = pickle.load(f)

            added_count = 0
            for schedule in seeded_population:
                if self._add_to_population(schedule):
                    added_count += 1

            self.logger.info("Loaded %d unique individuals from seed file.", added_count)
        except (OSError, pickle.UnpicklingError, KeyError, EOFError):
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
                self._notify_on_finish(self.population, self.pareto_front())
                return True
        except Exception:
            self.logger.exception("An error occurred during the genetic algorithm run.")
            return False
        except KeyboardInterrupt:
            self.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            self.nsga2.non_dominated_sort(self.population)
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
        builder = ScheduleBuilder(self.team_factory, self.event_factory, self.config, seeder)

        num_created = 0
        while len(self.population) < num_to_create and attempts < max_attempts:
            builder.rng = Random(seeder.randint(*RANDOM_SEED))
            schedule = builder.build()
            if self.repairer.repair(schedule) and self._add_to_population(schedule):
                schedule.fitness = self.evaluator.evaluate(schedule)
                num_created += 1
            else:
                attempts += 1
                self.logger.info("Attempt %d: Created %d valid schedules so far.", attempts, num_created)

        if not self.population or not self.population[0].fitness:
            self.logger.critical("No valid initial schedules could be built.")
            return

        self.nsga2.non_dominated_sort(self.population)
        self._cleanup_population_tracking()
        self._update_fitness_history()

    def generation(self) -> bool:
        """Perform a single generation step of the genetic algorithm."""
        num_offspring = self.ga_params.population_size - self.ga_params.elite_size
        for generation in range(self.ga_params.generations):
            self.evolve(num_offspring)
            self.nsga2.non_dominated_sort(self.population)
            self.population = list(self.elitism.select(self.population, self.ga_params.population_size))
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

            p1, p2 = s.select(self.population, 2)
            if p1 == p2:
                continue

            for child in self.crossover_child([p1, p2]):
                self.mutate_child(child)
                if self.repairer.repair(child) and self._add_to_population(child):
                    child.fitness = self.evaluator.evaluate(child)
                    child_count += 1

                if child_count >= num_offspring:
                    break

    def crossover_child(self, parents: list[Schedule, Schedule]) -> Iterator[Schedule]:
        """Evolve the population by one individual and return the best of the parents and child."""
        crossover_chance = self.ga_params.crossover_chance
        crossover_success = False
        if crossover_chance > self.rng.random():
            for child in self.rng.choice(self.crossovers).crossover(parents):
                crossover_success = True
                yield child
            self._crossover_ratio["success" if crossover_success else "failure"] += 1

    def mutate_child(self, child: Schedule) -> None:
        """Mutate the child schedule."""
        low = self.ga_params.mutation_chance_low
        high = self.ga_params.mutation_chance_high
        mutation_chance = (high - low) // 2

        if mutation_chance > self.rng.random():
            if mutation_success := self.rng.choice(self.mutations).mutate(child):
                pass
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
        c_success = self._crossover_ratio["success"]
        c_total = c_success + self._crossover_ratio["failure"]
        c_success_percentage = c_success / c_total if c_total > 0 else 0.0

        m_success = self._mutation_ratio["success"]
        m_total = m_success + self._mutation_ratio["failure"]
        m_success_percentage = m_success / m_total if m_total > 0 else 0.0

        self.logger.info(
            "Crossover success ratio: %s/%s = %s | %s",
            c_success,
            c_total,
            f"{c_success_percentage:.2%}",
            self._crossover_ratio,
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
