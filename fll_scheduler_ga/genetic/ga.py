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
from .island import Island
from .schedule import Population

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

    islands: list[Island] = field(default_factory=list, init=False, repr=False)

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
        self.islands = [
            Island(
                i,
                self.ga_params,
                self.config,
                self.rng,
                self.event_factory,
                self.team_factory,
                self.selections,
                self.crossovers,
                self.mutations,
                self.logger,
                self.evaluator,
                self.repairer,
            )
            for i in range(1, self.ga_params.num_islands + 1)
        ]

    def set_seed_file(self, file_path: str | Path | None) -> None:
        """Set the file path for loading a seed population."""
        if file_path:
            self._seed_file = Path(file_path)

    def pareto_front(self) -> Population:
        """Get the Pareto front for each island in the population."""
        if not self.population:
            return [p for i in self.islands for p in i.pareto_front()]
        return [p for p in self.population if p.rank == 0]

    def _calculate_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        front = self.pareto_front()
        if not front:
            return ()

        avg_fitness_front1 = defaultdict(int)

        for p in front:
            for i, _ in enumerate(self.evaluator.objectives):
                avg_fitness_front1[i] += p.fitness[i]

        return tuple(s / len(front) for s in avg_fitness_front1.values())

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history.append(self._calculate_this_gen_fitness())

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time.time()
        self._notify_on_start()

        try:
            self.initialize_population()
            if not any(self.islands[i].population for i in range(self.ga_params.num_islands)):
                self.logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
            self.finalize()
            self._notify_on_finish()
        except Exception:
            self.logger.exception("An error occurred during the genetic algorithm run.")
            self.finalize()
            self.update_fitness_history()
            self._notify_on_finish()
            return False
        except KeyboardInterrupt:
            self.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            self.finalize()
            self.update_fitness_history()
            self._notify_on_finish()
            return True
        finally:
            if start_time:
                self.logger.info("Total time taken: %.2f seconds", time.time() - start_time)
        return True

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seeded_population = []
        if self._seed_file and self._seed_file.exists():
            seeded_population = self.retrieve_seed_population()

        if seeded_population:
            for i, schedule in enumerate(seeded_population):
                island_idx = i % self.ga_params.num_islands
                self.islands[island_idx].add_to_population(schedule)

        for i in range(self.ga_params.num_islands):
            self.islands[i].initialize()

        self.update_fitness_history()

    def retrieve_seed_population(self) -> Population | None:
        """Load and integrate a population from a seed file."""
        self.logger.info("Loading seed population from: %s", self._seed_file)
        try:
            with shelve.open(self._seed_file) as shelf:
                shelf_config: TournamentConfig = shelf.get("config")
                if self.config.num_teams != shelf_config.num_teams or self.config.rounds != shelf_config.rounds:
                    self.logger.warning(
                        "Seed file configuration does not match current configuration. Using current config."
                    )
                    return None
                return shelf.get("population", [])
        except (OSError, KeyError, EOFError, Exception):
            self.logger.exception("Could not load or parse seed file. Starting with a fresh population.")

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        num_islands = self.ga_params.num_islands
        migration_size = self.ga_params.migration_size
        for generation in range(self.ga_params.generations):
            for i in range(num_islands):
                ratios = self.islands[i].evolve()
                self._offspring_ratio.update(ratios["offspring"])
                self._mutation_ratio.update(ratios["mutation"])

            self.update_fitness_history()
            self._notify_on_generation_end(generation)

            if num_islands > 1 and migration_size > 0 and (generation + 1) % self.ga_params.migration_interval == 0:
                self.migrate(num_islands)

    def migrate(self, num_islands: int) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        all_migrants = (self.islands[i].get_migrants() for i in range(num_islands))

        for i, migrants in enumerate(all_migrants):
            dest_i = (i - 1) % num_islands
            self.islands[dest_i].receive_migrants(migrants)

    def finalize(self) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        unique_pop = list({ind for island in self.islands for ind in island.population})

        for ind in unique_pop:
            if ind.fitness is None:
                ind.fitness = self.evaluator.evaluate(ind)

        self.logger.info(
            "Aggregated %d islands into a population of %d unique individuals for final selection.",
            len(self.islands),
            len(unique_pop),
        )

        self.population = sorted(unique_pop, key=lambda s: (s.rank, -sum(s.fitness)))

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
