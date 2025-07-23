"""Genetic algorithm for FLL Scheduler GA."""

import logging
import shelve
import time
from collections import Counter, defaultdict
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

    island_hash_to_pop: dict[int, dict[int, Schedule]] = field(default_factory=dict, init=False, repr=False)

    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _mutation_ratio: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        self.builder = ScheduleBuilder(
            self.team_factory,
            self.event_factory,
            self.config,
            Random(seeder.randint(*RANDOM_SEED)),
        )
        self.repairer.rng = Random(seeder.randint(*RANDOM_SEED))
        self.nsga3 = NSGA3(self.rng, len(self.evaluator.objectives), self.ga_params.population_size)
        self.island_hash_to_pop = {i: {} for i in range(self.ga_params.num_islands)}

    def set_seed_file(self, file_path: str | Path | None) -> None:
        """Set the file path for loading a seed population."""
        if file_path:
            self._seed_file = Path(file_path)

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        if not self.population:
            return [p for island in self.island_hash_to_pop.values() for p in island.values() if p.rank == 0]
        return [p for p in self.population if p.rank == 0]

    def _calculate_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        front = [p for island in self.island_hash_to_pop.values() for p in island.values() if p.rank == 0]
        if not front:
            return ()

        avg_fitness_front1 = defaultdict(list)

        for p in front:
            for i, _ in enumerate(self.evaluator.objectives):
                avg_fitness_front1[i].append(p.fitness[i])

        return tuple(sum(s) / len(s) for s in avg_fitness_front1.values())

    def update_fitness_history(self) -> tuple[float, ...]:
        """Update the fitness history with the current generation's fitness."""
        if this_gen_fitness := self._calculate_this_gen_fitness():
            self.fitness_history.append(this_gen_fitness)
        return this_gen_fitness

    def _add_to_island_population(self, schedule: Schedule, island_idx: int) -> bool:
        """Add a schedule to a specific island's population if it's not a duplicate."""
        island_hash_to_pop = self.island_hash_to_pop[island_idx]
        schedule_hash = hash(schedule)
        if schedule_hash not in island_hash_to_pop:
            island_hash_to_pop[schedule_hash] = schedule
            return True
        return False

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time.time()
        self._notify_on_start()

        try:
            self.initialize_population()
            if not self.island_hash_to_pop:
                self.logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
            self.aggregate_and_finalize_population()
            self._notify_on_finish()
        except Exception:
            self.logger.exception("An error occurred during the genetic algorithm run.")
            return False
        except KeyboardInterrupt:
            self.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            self.aggregate_and_finalize_population()
            self.update_fitness_history()
            self._notify_on_finish()
            return True
        finally:
            if start_time:
                self.logger.info("Total time taken: %.2f seconds", time.time() - start_time)
        return True

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        if self._seed_file and self._seed_file.exists():
            self._load_population_from_seed()

        for i in range(self.ga_params.num_islands):
            self._populate_island(i)
            self.island_hash_to_pop[i] = {hash(s): s for s in self.nsga3.select(self.island_hash_to_pop[i].values())}

        self.update_fitness_history()

    def _load_population_from_seed(self) -> None:
        """Load and integrate a population from a seed file."""
        self.logger.info("Loading seed population from: %s", self._seed_file)
        try:
            with shelve.open(self._seed_file) as shelf:
                shelf_config: TournamentConfig = shelf.get("config")
                if self.config.num_teams != shelf_config.num_teams or self.config.rounds != shelf_config.rounds:
                    self.logger.warning(
                        "Seed file configuration does not match current configuration. Using current config."
                    )
                    return
                seeded_population: Population = shelf.get("population", [])

            for i, schedule in enumerate(seeded_population):
                island_idx = i % self.ga_params.num_islands
                if self._add_to_island_population(schedule, island_idx) and schedule.fitness is None:
                    schedule.fitness = self.evaluator.evaluate(schedule)
        except (OSError, KeyError, EOFError, Exception):
            self.logger.exception("Could not load or parse seed file. Starting with a fresh population.")

    def _populate_island(self, island_idx: int) -> None:
        """Populate a single island with random organisms."""
        # If the island already has enough individuals from seed file, skip population.
        num_to_create = self.ga_params.population_size - len(self.island_hash_to_pop[island_idx])
        if num_to_create <= 0:
            return

        self.logger.info("Initializing island %d with %d individuals.", island_idx, num_to_create)
        seeder = Random(self.rng.randint(*RANDOM_SEED))
        attempts, max_attempts = ATTEMPTS
        num_created = 0
        while len(self.island_hash_to_pop[island_idx]) < self.ga_params.population_size and attempts < max_attempts:
            self.builder.rng = Random(seeder.randint(*RANDOM_SEED))
            schedule = self.builder.build()
            if self.repairer.repair(schedule) and self._add_to_island_population(schedule, island_idx):
                schedule.fitness = self.evaluator.evaluate(schedule)
                num_created += 1
                max_attempts += 1
            else:
                attempts += 1

        if num_created < num_to_create:
            self.logger.warning(
                "Island %d: only created %d/%d valid individuals.", island_idx, num_created, num_to_create
            )

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        num_islands = self.ga_params.num_islands
        for generation in range(self.ga_params.generations):
            for i in range(num_islands):
                self._evolve_island(i)
                self.island_hash_to_pop[i] = {
                    hash(s): s for s in self.nsga3.select(self.island_hash_to_pop[i].values())
                }

            self.update_fitness_history()
            self._notify_on_generation_end(generation)

            if num_islands > 1 and (generation + 1) % self.ga_params.migration_interval == 0:
                self._migrate()

    def _evolve_island(self, island_idx: int) -> None:
        """Evolve an island's population to create a new generation."""
        island_pop = list(self.island_hash_to_pop[island_idx].values())
        if not island_pop:
            return

        num_offspring = self.ga_params.population_size - self.ga_params.elite_size
        child_count = 0

        repair = self.repairer.repair
        evaluate = self.evaluator.evaluate
        offspring_ratio = self._offspring_ratio

        # while child_count < num_offspring:
        for _ in range(num_offspring):
            parents = tuple(self.rng.choice(self.selections).select(island_pop, num_parents=2))
            if len(set(parents)) < 2:
                continue

            if self.ga_params.crossover_chance < self.rng.random():
                continue

            for child in self.rng.choice(self.crossovers).crossover(parents):
                if self.ga_params.mutation_chance > self.rng.random():
                    mutation_success = self.rng.choice(self.mutations).mutate(child)
                    self._mutation_ratio["success" if mutation_success else "failure"] += 1

                if repair(child) and self._add_to_island_population(child, island_idx):
                    child.fitness = evaluate(child)
                    child_count += 1
                    offspring_ratio["success"] += 1
                else:
                    offspring_ratio["failure"] += 1

                if child_count >= num_offspring:
                    break

    def _migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        num_islands = self.ga_params.num_islands
        migration_size = self.ga_params.migration_size
        if migration_size == 0:
            return

        # Receive migrants using a ring topology (island i receives from neighboring island)
        for i in range(num_islands):
            src_i = (i - 1) % num_islands
            src_island = sorted(self.island_hash_to_pop[src_i].values(), key=lambda s: (s.rank, -sum(s.fitness)))
            for migrant in src_island[:migration_size]:
                self._add_to_island_population(migrant, i)

    def aggregate_and_finalize_population(self) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        self.population = list({ind for island in self.island_hash_to_pop.values() for ind in island.values()})
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self.evaluator.evaluate(ind)

        self.logger.info(
            "Aggregated %d islands into a population of %d unique individuals for final selection.",
            len(self.island_hash_to_pop),
            len(self.population),
        )

        self.population.sort(key=lambda s: (s.rank, -sum(s.fitness)))

    def _notify_on_start(self) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(self.ga_params.generations)

    def _notify_on_generation_end(self, generation: int) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(
                generation + 1,
                self.ga_params.generations,
                self.ga_params.population_size * self.ga_params.num_islands,
                self.fitness_history[-1] if self.fitness_history else (),
                len(self.pareto_front()),
            )

    def _notify_on_finish(self) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        o_success = self._offspring_ratio["success"]
        o_total = o_success + self._offspring_ratio["failure"]
        o_success_percentage = o_success / o_total if o_total > 0 else 0.0

        m_success = self._mutation_ratio["success"]
        m_total = m_success + self._mutation_ratio["failure"]
        m_success_percentage = m_success / m_total if m_total > 0 else 0.0

        self.logger.info(
            "Offspring success ratio: %s/%s = %s\n\t%s",
            o_success,
            o_total,
            f"{o_success_percentage:.2%}",
            self._offspring_ratio,
        )
        self.logger.info(
            "Mutation success ratio: %s/%s = %s\n\t%s",
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
            obs.on_finish(self.population, self.pareto_front())
