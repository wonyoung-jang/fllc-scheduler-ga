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
                self.builder,
                self.nsga3,
            )
            for i in range(1, self.ga_params.num_islands + 1)
        ]

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

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
        this_gen_fitness = self._calculate_this_gen_fitness()
        if self.fitness_history and self.fitness_history[-1] <= this_gen_fitness:
            self.ga_params.mutation_chance *= 0.9
        else:
            self.ga_params.mutation_chance *= 1.1
        self.fitness_history.append(this_gen_fitness)

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
        except Exception:
            self.logger.exception("An error occurred during the genetic algorithm run.")
            self.update_fitness_history()
            return False
        except KeyboardInterrupt:
            self.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            self.update_fitness_history()
            return True
        finally:
            if start_time:
                self.logger.info("Total time taken: %.2f seconds", time.time() - start_time)
            self.finalize()
            self._notify_on_finish()
        return True

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seed_path = self._seed_file
        if seed_path and seed_path.exists() and (seed_pop := self.retrieve_seed_population(seed_path)):
            seed_pop.sort(key=lambda s: (s.rank, -sum(s.fitness)))
            for i, schedule in enumerate(seed_pop):
                island_idx = i % self.ga_params.num_islands
                self.islands[island_idx].add_to_population(schedule)

        self.logger.info("Initializing %d islands...", self.ga_params.num_islands)
        for i in range(self.ga_params.num_islands):
            self.islands[i].initialize()

    def retrieve_seed_population(self, seed_path: Path) -> Population | None:
        """Load and integrate a population from a seed file."""
        self.logger.info("Loading seed population from: %s", seed_path)
        try:
            with shelve.open(seed_path) as shelf:
                seed_config: TournamentConfig = shelf.get("config", None)
                num_teams_changed = self.config.num_teams != seed_config.num_teams
                config_changed = self.config.rounds != seed_config.rounds
                if num_teams_changed or config_changed:
                    self.logger.warning("Seed population does not match current config. Using current...")
                    return None
                return shelf.get("population", [])
        except (OSError, KeyError, EOFError, Exception):
            self.logger.exception("Could not load or parse seed file. Starting with a fresh population.")

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        num_islands = self.ga_params.num_islands
        migration_size = self.ga_params.migration_size
        migration_interval = self.ga_params.migration_interval
        offspring_ratio = self._offspring_ratio
        mutation_ratio = self._mutation_ratio

        for generation in range(self.ga_params.generations):
            for i in range(num_islands):
                ratios = self.islands[i].evolve()
                offspring_ratio.update(ratios["offspring"])
                mutation_ratio.update(ratios["mutation"])

            self.update_fitness_history()
            self._notify_on_generation_end(generation)

            if num_islands <= 1 or migration_size <= 0:
                continue

            if (generation + 1) % migration_interval == 0:
                self.migrate(num_islands, migration_size)

    def migrate(self, num_islands: int, migration_size: int) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        all_migrants = (island.get_migrants(migration_size) for island in self.islands)

        for i, migrants in enumerate(all_migrants):
            dest_i = (i + 1) % num_islands
            self.islands[dest_i].receive_migrants(migrants)

    def finalize(self) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        unique_pop = list({ind for island in self.islands for ind in island.finalize_island()})

        self.logger.info(
            "Finalized %d islands with population of %d unique individuals.",
            len(self.islands),
            len(unique_pop),
        )

        self.population = sorted(
            self.nsga3.select(
                unique_pop,
                pop_size=len(unique_pop),
            ),
            key=lambda s: (
                s.rank,
                -sum(s.fitness),
            ),
        )

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
        o_percent = o_success / o_total if o_total > 0 else 0.0

        m_success = self._mutation_ratio["success"]
        m_total = m_success + self._mutation_ratio["failure"]
        m_percent = m_success / m_total if m_total > 0 else 0.0

        self.logger.info(
            "Offspring success ratio: %s/%s = %s\n\t%s",
            o_success,
            o_total,
            f"{o_percent:.2%}",
            self._offspring_ratio,
        )
        self.logger.info(
            "Mutation success ratio: %s/%s = %s\n\t%s",
            m_success,
            m_total,
            f"{m_percent:.2%}",
            self._mutation_ratio,
        )
        self.logger.info(
            "Unique/Total individuals: %s/%s",
            len({hash(s) for s in self.population}),
            self.ga_params.population_size * self.ga_params.num_islands,
        )
        for obs in self.observers:
            obs.on_finish(self.population, self.pareto_front())
