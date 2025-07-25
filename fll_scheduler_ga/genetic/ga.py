"""Genetic algorithm for FLL Scheduler GA."""

import shelve
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.config import TournamentConfig

from ..config.constants import RANDOM_SEED_RANGE
from ..observers.base_observer import GaObserver
from .builder import ScheduleBuilder
from .ga_context import GaContext
from .island import Island
from .schedule import Population


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    rng: Random
    observers: tuple[GaObserver]

    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    fitness_improvement_history: list[bool] = field(default_factory=list, init=False, repr=False)
    total_population: Population = field(default_factory=list, init=False, repr=False)
    islands: list[Island] = field(init=False, repr=False)

    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _mutation_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _expected_population_size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._expected_population_size = self.context.ga_params.population_size * self.context.ga_params.num_islands
        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        builder = ScheduleBuilder(
            self.context.team_factory,
            self.context.event_factory,
            self.context.config,
            Random(seeder.randint(*RANDOM_SEED_RANGE)),
        )
        self.context.repairer.rng = Random(seeder.randint(*RANDOM_SEED_RANGE))
        self.islands = [
            Island(
                i,
                Random(seeder.randint(*RANDOM_SEED_RANGE)),
                builder,
                self.context,
            )
            for i in range(1, self.context.ga_params.num_islands + 1)
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
        if not self.total_population:
            return [p for i in self.islands for p in i.pareto_front()]
        return [p for p in self.total_population if p.rank == 0]

    def _calculate_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        if not (front := self.pareto_front()):
            return ()

        avg_fitness_front1 = defaultdict(int)

        for p in front:
            for i, _ in enumerate(self.context.evaluator.objectives):
                avg_fitness_front1[i] += p.fitness[i]

        return tuple(s / len(front) for s in avg_fitness_front1.values())

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        this_gen_fitness = self._calculate_this_gen_fitness()
        crossover_low = 0.3
        crossover_high = 0.9
        mutation_low = 0.01
        mutation_high = 0.5

        if self.fitness_history and self.fitness_history[-1] < this_gen_fitness:
            self.fitness_improvement_history.append(True)
        else:
            self.fitness_improvement_history.append(False)

        if len(self.fitness_improvement_history) >= 5:
            last_five_improvements = self.fitness_improvement_history[-5:]
            improved_count = last_five_improvements.count(True)

            # 1/5 generations improved -> reduce mutation chance
            if improved_count < 1:
                self.context.ga_params.crossover_chance = max(
                    crossover_low,
                    self.context.ga_params.crossover_chance - 0.1,
                )
                self.context.ga_params.mutation_chance = max(
                    mutation_low,
                    self.context.ga_params.mutation_chance - 0.01,
                )
            # More than 1/5 generations improved -> increase mutation chance, converging too early
            elif improved_count > 1:
                self.context.ga_params.crossover_chance = min(
                    crossover_high,
                    self.context.ga_params.crossover_chance + 0.1,
                )
                self.context.ga_params.mutation_chance = min(
                    mutation_high,
                    self.context.ga_params.mutation_chance + 0.01,
                )

        self.fitness_history.append(this_gen_fitness)

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time.time()
        self._notify_on_start(self.context.ga_params.generations)

        try:
            self.initialize_population()
            if not any(self.islands[i].population for i in range(self.context.ga_params.num_islands)):
                self.context.logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
        except Exception:
            self.context.logger.exception("An error occurred during the genetic algorithm run.")
            self.update_fitness_history()
            return False
        except KeyboardInterrupt:
            self.context.logger.warning("Genetic algorithm run interrupted by user. Saving...")
            self.update_fitness_history()
            return True
        finally:
            if start_time:
                self.context.logger.info("Total time taken: %.2f seconds", time.time() - start_time)
            self.finalize()
            self._notify_on_finish(
                self._expected_population_size,
                self.total_population,
                self.pareto_front(),
                self._mutation_ratio,
                self._offspring_ratio,
            )
        return True

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seed_path = self._seed_file
        if seed_path and seed_path.exists() and (seed_pop := self.retrieve_seed_population(seed_path)):
            seed_pop.sort(key=lambda s: (s.rank, -sum(s.fitness)))
            for i, schedule in enumerate(seed_pop):
                island_idx = i % self.context.ga_params.num_islands
                self.islands[island_idx].add_to_population(schedule)

        self.context.logger.info("Initializing %d islands...", self.context.ga_params.num_islands)
        for i in range(self.context.ga_params.num_islands):
            self.islands[i].initialize()

    def retrieve_seed_population(self, seed_path: Path) -> Population | None:
        """Load and integrate a population from a seed file."""
        self.context.logger.info("Loading seed population from: %s", seed_path)
        try:
            with shelve.open(seed_path) as shelf:
                seed_config: TournamentConfig = shelf.get("config", None)
                num_teams_changed = self.context.config.num_teams != seed_config.num_teams
                config_changed = self.context.config.rounds != seed_config.rounds
                if num_teams_changed or config_changed:
                    self.context.logger.warning("Seed population does not match current config. Using current...")
                    return None
                return shelf.get("population", [])
        except (OSError, KeyError, EOFError, Exception):
            self.context.logger.exception("Could not load or parse seed file. Starting with a fresh population.")

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        num_islands = self.context.ga_params.num_islands
        migration_size = self.context.ga_params.migration_size
        migration_interval = self.context.ga_params.migration_interval
        num_generations = self.context.ga_params.generations
        offspring_ratio = self._offspring_ratio
        mutation_ratio = self._mutation_ratio
        expected_pop_size = self._expected_population_size

        for generation in range(num_generations):
            for i in range(num_islands):
                ratios = self.islands[i].evolve()
                offspring_ratio.update(ratios["offspring"])
                mutation_ratio.update(ratios["mutation"])

            self.update_fitness_history()
            self._notify_on_generation_end(
                generation + 1,
                num_generations,
                expected_pop_size,
                self.fitness_history[-1] if self.fitness_history else (),
                len(self.pareto_front()),
            )

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

        self.context.logger.info(
            "Finalized %d islands with population of %d unique individuals.",
            len(self.islands),
            len(unique_pop),
        )

        self.total_population = sorted(
            self.context.nsga3.select(
                unique_pop,
                population_size=len(unique_pop),
            ),
            key=lambda s: (
                s.rank,
                -sum(s.fitness),
            ),
        )

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_on_generation_end(
        self,
        generation: int,
        num_generations: int,
        expected: int,
        fitness: tuple[float, ...],
        pareto_size: int,
    ) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(generation, num_generations, expected, fitness, pareto_size)

    def _notify_on_finish(
        self,
        expected: int,
        pop: Population,
        pareto_front: Population,
        mutation_ratio: Counter | None = None,
        offspring_ratio: Counter | None = None,
    ) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        o_success = offspring_ratio["success"]
        o_total = o_success + offspring_ratio["failure"]
        o_percent = f"{o_success / o_total if o_total > 0 else 0.0:.2%}"

        m_success = mutation_ratio["success"]
        m_total = m_success + mutation_ratio["failure"]
        m_percent = f"{m_success / m_total if m_total > 0 else 0.0:.2%}"

        self.context.logger.info("Offspring success: %s/%s = %s", o_success, o_total, o_percent)
        self.context.logger.info("Mutation success: %s/%s = %s", m_success, m_total, m_percent)
        self.context.logger.info("Unique/Total individuals: %s/%s", len(pop), expected)

        for obs in self.observers:
            obs.on_finish(pop, pareto_front)
