"""Genetic algorithm for FLL Scheduler GA."""

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.config import TournamentConfig

from ..config.constants import EPSILON, RANDOM_SEED_RANGE
from ..data_model.schedule import Population
from ..observers.base_observer import GaObserver
from .builder import ScheduleBuilder
from .ga_context import GaContext
from .island import Island


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    rng: Random
    observers: tuple[GaObserver]

    fitness_history: list[tuple] = field(default_factory=list, init=False, repr=False)
    fitness_improvement_history: list[bool] = field(default_factory=list, init=False, repr=False)
    total_population: Population = field(default_factory=list, init=False, repr=False)
    islands: list[Island] = field(default_factory=list, init=False, repr=False)

    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _crossover_ratio: dict = field(default_factory=dict, init=False, repr=False)
    _mutation_ratio: dict = field(default_factory=dict, init=False, repr=False)
    _c_base: float = field(init=False, repr=False)
    _m_base: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._c_base = self.context.ga_params.crossover_chance
        self._m_base = self.context.ga_params.mutation_chance
        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        builder = ScheduleBuilder(
            self.context.team_factory,
            self.context.event_factory,
            self.context.config,
            Random(seeder.randint(*RANDOM_SEED_RANGE)),
        )
        self.context.repairer.rng = Random(seeder.randint(*RANDOM_SEED_RANGE))
        self.islands.extend(
            Island(
                i,
                Random(seeder.randint(*RANDOM_SEED_RANGE)),
                builder,
                self.context,
            )
            for i in range(1, self.context.ga_params.num_islands + 1)
        )

        self._crossover_ratio["success"] = Counter()
        self._crossover_ratio["total"] = Counter()
        self._crossover_ratio["ratio"] = {}
        for crossover in self.context.crossovers:
            self._crossover_ratio["success"][f"{crossover!s}"] = 0
            self._crossover_ratio["total"][f"{crossover!s}"] = 0
            self._crossover_ratio["ratio"][f"{crossover!s}"] = 0.0

        self._mutation_ratio["success"] = Counter()
        self._mutation_ratio["total"] = Counter()
        self._mutation_ratio["ratio"] = {}
        for mutation in self.context.mutations:
            self._mutation_ratio["success"][f"{mutation!s}"] = 0
            self._mutation_ratio["total"][f"{mutation!s}"] = 0
            self._mutation_ratio["ratio"][f"{mutation!s}"] = 0.0

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

    def _get_this_gen_fitness(self) -> tuple[float, ...]:
        """Calculate the average fitness of the current generation."""
        island_last_gens = (i.get_last_gen_fitness() for i in self.islands)
        return tuple(sum(s) / len(self.islands) for s in zip(*island_last_gens, strict=True))

    def _get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def adapt_operator_probabilities(self) -> None:
        """Adapt the operator probabilities based on the fitness history."""
        c_chance = self.context.ga_params.crossover_chance
        m_chance = self.context.ga_params.mutation_chance
        len_improvements = len(self.fitness_improvement_history)

        if len_improvements >= 10:
            last_improvements = self.fitness_improvement_history[-len_improvements // 5 :]
            improved_ratio = last_improvements.count(1) / len(last_improvements)

            # Less than 1/5 generations improved -> decrease operator chance / exploit
            if improved_ratio < 0.2:
                self.context.ga_params.crossover_chance = max(0.01, c_chance * EPSILON)
                self.context.ga_params.mutation_chance = max(0.001, m_chance * EPSILON)
                self.context.logger.debug(
                    "Reduced crossover chance to %.2f and mutation chance to %.2f",
                    self.context.ga_params.crossover_chance,
                    self.context.ga_params.mutation_chance,
                )
            # More than 1/5 generations improved -> increase operator chance / explore
            elif improved_ratio > 0.2:
                self.context.ga_params.crossover_chance = min(0.9999, c_chance / EPSILON)
                self.context.ga_params.mutation_chance = min(0.9999, m_chance / EPSILON)
                self.context.logger.debug(
                    "Increased crossover chance to %.2f and mutation chance to %.2f",
                    self.context.ga_params.crossover_chance,
                    self.context.ga_params.mutation_chance,
                )
            else:
                self.context.ga_params.crossover_chance = self._c_base
                self.context.ga_params.mutation_chance = self._m_base
                self.context.logger.debug(
                    "No change in crossover or mutation chance, current values: %.2f, %.2f",
                    self.context.ga_params.crossover_chance,
                    self.context.ga_params.mutation_chance,
                )

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        this_gen_fitness = self._get_this_gen_fitness()
        last_gen_fitness = self._get_last_gen_fitness()

        if self.fitness_history:
            if sum(last_gen_fitness) < sum(this_gen_fitness):
                self.fitness_improvement_history.append(1)
            elif sum(last_gen_fitness) > sum(this_gen_fitness):
                self.fitness_improvement_history.append(-1)
            else:
                self.fitness_improvement_history.append(0)

        self.adapt_operator_probabilities()
        self.fitness_history.append(this_gen_fitness)

    def run(self) -> bool:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time()
        self._notify_on_start(self.context.ga_params.generations)

        try:
            self.initialize_population()
            if not any(self.islands[i].selected for i in range(self.context.ga_params.num_islands)):
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
            self.context.logger.info("Total time taken: %.2f seconds", time() - start_time)
            self.finalize()
            self._notify_on_finish(self.total_population, self.pareto_front())
        return True

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seed_path = self._seed_file
        if seed_path and seed_path.exists() and (seed_pop := self.retrieve_seed_population()):
            seed_pop.sort(key=lambda s: (s.rank, -sum(s.fitness)))
            for spi, schedule in enumerate(seed_pop):
                i = spi % self.context.ga_params.num_islands
                self.islands[i].add_to_population(schedule)

        self.context.logger.info("Initializing %d islands...", self.context.ga_params.num_islands)
        for i in range(self.context.ga_params.num_islands):
            self.islands[i].initialize()

    def retrieve_seed_population(self) -> Population | None:
        """Load and integrate a population from a seed file."""
        self.context.logger.info("Loading seed population from: %s", self._seed_file)
        try:
            with self._seed_file.open("rb") as f:
                seed_data = pickle.load(f)
                seed_config: TournamentConfig = seed_data["config"]
                num_teams_changed = self.context.config.num_teams != seed_config.num_teams
                config_changed = self.context.config.rounds != seed_config.rounds
                if num_teams_changed or config_changed:
                    self.context.logger.warning("Seed population does not match current config. Using current...")
                    return None
                return seed_data["population"]
        except (OSError, pickle.PicklingError):
            self.context.logger.exception("Could not load or parse seed file. Starting with a fresh population.")
        except EOFError:
            self.context.logger.debug("Pickle file is empty")

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for generation in range(self.context.ga_params.generations):
            if self._migration_condition(generation):
                self.migrate()

            for i in range(self.context.ga_params.num_islands):
                self.islands[i].handle_underpopulation()
                self.islands[i].evolve()
                self.islands[i].update_fitness_history()

            self._notify_on_generation_end(
                generation + 1,
                self.context.ga_params.generations,
                len(self),
                self.fitness_history[-1] if self.fitness_history else (),
                len(self.pareto_front()),
            )

            self.update_fitness_history()

    def _migration_condition(self, generation: int) -> bool:
        """Check if migration should occur based on the generation and migration interval."""
        return (
            self.context.ga_params.num_islands > 1
            and self.context.ga_params.migration_size > 0
            and (generation + 1) % self.context.ga_params.migration_interval == 0
        )

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        all_migrants = (
            (
                i,
                island.get_migrants(
                    self.context.ga_params.migration_size,
                ),
            )
            for i, island in enumerate(self.islands)
        )

        for i, migrants in all_migrants:
            dest_i = (i + 1) % self.context.ga_params.num_islands
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
            self.context.nsga3.select(unique_pop, population_size=len(unique_pop)).values(),
            key=lambda s: (s.rank, -sum(s.fitness)),
        )

        for i in range(self.context.ga_params.num_islands):
            c_ratio = self.islands[i].crossover_ratio
            m_ratio = self.islands[i].mutation_ratio
            o_ratio = self.islands[i].offspring_ratio
            for tracker, crossovers in c_ratio.items():
                for crossover, count in crossovers.items():
                    self._crossover_ratio[tracker][crossover] += count
            for tracker, mutations in m_ratio.items():
                for mutation, count in mutations.items():
                    self._mutation_ratio[tracker][mutation] += count
            for tracker, count in o_ratio.items():
                self._offspring_ratio[tracker] += count

        # Log final crossover statistics
        crossover_log = f"{'=' * 20}\nCrossover statistics"
        for success, total, crossover in zip(
            self._crossover_ratio["success"].values(),
            self._crossover_ratio["total"].values(),
            self.context.crossovers,
            strict=False,
        ):
            self._crossover_ratio["ratio"][f"{crossover!s}"] += success / total if total > 0 else 0.0
        for tracker, crossovers in self._crossover_ratio.items():
            crossover_log += f"\n  {tracker}:"
            for crossover, count in sorted(crossovers.items()):
                crossover_log += f"\n    Crossover {crossover}: {count}"
        self.context.logger.info(crossover_log)

        # Log final mutation statistics
        mutation_log = f"{'=' * 20}\nMutation statistics"
        for success, total, mutation in zip(
            self._mutation_ratio["success"].values(),
            self._mutation_ratio["total"].values(),
            self.context.mutations,
            strict=False,
        ):
            self._mutation_ratio["ratio"][f"{mutation!s}"] += success / total if total > 0 else 0.0
        for tracker, mutations in self._mutation_ratio.items():
            mutation_log += f"\n  {tracker}:"
            for mutation, count in sorted(mutations.items()):
                mutation_log += f"\n    Mutation {mutation}: {count}"
        self.context.logger.info(mutation_log)

        self.context.logger.info(
            "Crossovers: %s/%s, Mutations: %s/%s",
            sum(self._crossover_ratio["success"].values()),
            sum(self._crossover_ratio["total"].values()),
            sum(self._mutation_ratio["success"].values()),
            sum(self._mutation_ratio["total"].values()),
        )

        o_success = self._offspring_ratio["success"]
        o_total = self._offspring_ratio["total"]
        o_percent = f"{o_success / o_total if o_total > 0 else 0.0:.2%}"
        offspring_log = (
            f"{'=' * 20}"
            f"\nOffspring statistics"
            f"\n  Successes: {o_success}"
            f"\n  Failures: {o_total - o_success}"
            f"\n  Total: {o_total}"
            f"\n  Success rate: {o_percent}"
        )
        self.context.logger.info(offspring_log)

        self.context.logger.info(
            "Unique/Total individuals: %s/%s",
            len(self.total_population),
            len(self),
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

    def _notify_on_finish(self, pop: Population, pareto_front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, pareto_front)
