"""Genetic algorithm for FLL Scheduler GA."""

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from time import time
from typing import TYPE_CHECKING

from ..config.constants import EPSILON, RANDOM_SEED_RANGE
from ..config.ga_context import GaContext
from ..config.ga_parameters import GaParameters
from ..data_model.schedule import Population
from ..observers.base_observer import GaObserver
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from .builder import ScheduleBuilder
from .island import Island

if TYPE_CHECKING:
    from ..config.config import TournamentConfig


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
    ga_params: GaParameters = field(init=False)

    _seed_file: Path | None = field(default=None, init=False, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, init=False, repr=False)
    _crossover_ratio: dict = field(default_factory=dict, init=False, repr=False)
    _mutation_ratio: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.ga_params = self.context.app_config.ga_params
        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        builder = ScheduleBuilder(
            self.context.team_factory,
            self.context.event_factory,
            self.context.app_config.tournament,
            Random(seeder.randint(*RANDOM_SEED_RANGE)),
        )

        self.context.repairer.rng = Random(seeder.randint(*RANDOM_SEED_RANGE))
        self.islands.extend(
            Island(
                i,
                Random(seeder.randint(*RANDOM_SEED_RANGE)),
                builder,
                self.context,
                self.ga_params,
            )
            for i in range(1, self.ga_params.num_islands + 1)
        )

        for tracker in ("success", "total"):
            self._crossover_ratio[tracker] = {str(c): 0 for c in self.context.crossovers}
            self._mutation_ratio[tracker] = {str(m): 0 for m in self.context.mutations}

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
        return tuple(
            sum(s) / len(self.islands)
            for s in zip(
                *(i.get_last_gen_fitness() for i in self.islands),
                strict=True,
            )
        )

    def _get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def adapt_operator_probabilities(self) -> None:
        """Adapt the operator probabilities based on the fitness history."""
        c_chance = self.ga_params.crossover_chance
        m_chance = self.ga_params.mutation_chance
        len_improvements = len(self.fitness_improvement_history)

        if len_improvements >= self.ga_params.migration_interval:
            last_improvements = self.fitness_improvement_history[-self.ga_params.migration_interval :]
            improved_ratio = last_improvements.count(1) / len(last_improvements)

            # Less than 1/5 generations improved -> decrease operator chance / exploit
            if improved_ratio < 0.2:
                self.ga_params.crossover_chance = max(0.001, c_chance * EPSILON)
                self.ga_params.mutation_chance = max(0.001, m_chance * EPSILON)
                self.context.logger.debug(
                    "Reduced crossover chance to %.2f and mutation chance to %.2f",
                    self.ga_params.crossover_chance,
                    self.ga_params.mutation_chance,
                )
            # More than 1/5 generations improved -> increase operator chance / explore
            elif improved_ratio > 0.2:
                self.ga_params.crossover_chance = min(0.999, c_chance / EPSILON)
                self.ga_params.mutation_chance = min(0.999, m_chance / EPSILON)
                self.context.logger.debug(
                    "Increased crossover chance to %.2f and mutation chance to %.2f",
                    self.ga_params.crossover_chance,
                    self.ga_params.mutation_chance,
                )

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        this_gen_fitness = self._get_this_gen_fitness()
        last_gen_fitness = self._get_last_gen_fitness()
        sum_this_gen_fitness = sum(this_gen_fitness) if this_gen_fitness else 0.0
        sum_last_gen_fitness = sum(last_gen_fitness) if last_gen_fitness else 0.0

        if self.fitness_history:
            if sum_last_gen_fitness < sum_this_gen_fitness:
                self.fitness_improvement_history.append(1)
            elif sum_last_gen_fitness > sum_this_gen_fitness:
                self.fitness_improvement_history.append(-1)
            else:
                self.fitness_improvement_history.append(0)

        self.adapt_operator_probabilities()
        self.fitness_history.append(this_gen_fitness)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time()
        self._notify_on_start(self.ga_params.generations)

        try:
            self.initialize_population()
            if not any(self.islands[i].selected for i in range(self.ga_params.num_islands)):
                self.context.logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
        except Exception:
            self.context.logger.exception("An error occurred during the genetic algorithm run.")
            self.update_fitness_history()
        except KeyboardInterrupt:
            self.context.logger.debug("Genetic algorithm run interrupted by user. Saving...")
            self.update_fitness_history()
        finally:
            self.context.logger.debug("Total time taken: %.2f seconds", time() - start_time)
            self.finalize()
            self._notify_on_finish(self.total_population, self.pareto_front())

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seed_path = self._seed_file
        if seed_path and seed_path.exists() and (seed_pop := self.retrieve_seed_population()):
            seed_pop.sort(key=lambda s: (s.rank, -sum(s.fitness)))
            for spi, schedule in enumerate(seed_pop):
                i = spi % self.ga_params.num_islands
                self.islands[i].add_to_population(schedule)

        self.context.logger.debug("Initializing %d islands...", self.ga_params.num_islands)
        for i in range(self.ga_params.num_islands):
            self.islands[i].initialize()

    def retrieve_seed_population(self) -> Population | None:
        """Load and integrate a population from a seed file."""
        self.context.logger.debug("Loading seed population from: %s", self._seed_file)
        try:
            with self._seed_file.open("rb") as f:
                seed_data = pickle.load(f)
                seed_config: TournamentConfig = seed_data["config"]
                num_teams_changed = self.context.app_config.tournament.num_teams != seed_config.num_teams
                config_changed = self.context.app_config.tournament.rounds != seed_config.rounds
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
        num_generations = self.ga_params.generations
        for generation in range(1, num_generations + 1):
            self.migrate(generation)
            self.run_single_epoch()
            self.update_fitness_history()
            self._notify_on_generation_end(
                generation=generation,
                num_generations=num_generations,
                expected=len(self),
                fitness=self._get_last_gen_fitness(),
                pareto_size=len(self.pareto_front()),
            )

    def run_single_epoch(self) -> None:
        """Run a single epoch of the genetic algorithm."""
        for island in self.islands:
            island.handle_underpopulation()
            island.evolve()
            island.handle_underpopulation()
            island.update_fitness_history()

    def migrate(self, generation: int) -> None:
        """Migrate the best individuals between islands using a random ring topology."""
        num_islands = self.ga_params.num_islands
        if not (
            num_islands > 1
            and self.ga_params.migration_size > 0
            and (generation) % self.ga_params.migration_interval == 0
        ):
            return

        r = self.rng.randrange(1, num_islands)  # Random offset
        for i, island in enumerate(self.islands):
            j = (i + r) % num_islands
            self.islands[j].receive_migrants(island.get_migrants())

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

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = list({ind for island in self.islands for ind in island.finalize_island()})
        self.total_population = sorted(
            self.context.nsga3.select(unique_pop, population_size=len(unique_pop)).values(),
            key=lambda s: (s.rank, -sum(s.fitness)),
        )

    def _aggregate_stats_from_islands(self) -> None:
        """Aggregate statistics from all islands."""
        for island in self.islands:
            for tracker, crossovers in island.crossover_ratio.items():
                for crossover, count in crossovers.items():
                    self._crossover_ratio[tracker][crossover] += count

            for tracker, mutations in island.mutation_ratio.items():
                for mutation, count in mutations.items():
                    self._mutation_ratio[tracker][mutation] += count

            for tracker, count in island.offspring_ratio.items():
                self._offspring_ratio[tracker] += count

    def _log_operators(self, name: str, ratios: dict[str, Counter], ops: tuple[Crossover | Mutation]) -> None:
        """Log statistics for crossover and mutation operators."""
        log = f"{name.capitalize()} statistics:"
        op_strings = [f"{op!s}" for op in ops]
        max_len = max((len(op) for op in op_strings), default=0) + 1
        for op in op_strings:
            success = ratios.get("success", {}).get(op, 0)
            total = ratios.get("total", {}).get(op, 0)
            ratio = success / total if total > 0 else 0.0
            log += f"\n  {op:<{max_len}}: {success}/{total} ({ratio:.2%})"
        self.context.logger.debug(log)

    def _log_aggregate_stats(self) -> None:
        """Log aggregate statistics across all islands."""
        final_log = f"{'=' * 20}\nFinal statistics"
        sum_crs_suc = sum(self._crossover_ratio.get("success", {}).values())
        sum_crs_tot = sum(self._crossover_ratio.get("total", {}).values())
        sum_crs_rte = f"{sum_crs_suc / sum_crs_tot if sum_crs_tot > 0 else 0.0:.2%}"
        sum_mut_suc = sum(self._mutation_ratio.get("success", {}).values())
        sum_mut_tot = sum(self._mutation_ratio.get("total", {}).values())
        sum_mut_rte = f"{sum_mut_suc / sum_mut_tot if sum_mut_tot > 0 else 0.0:.2%}"
        off_suc = self._offspring_ratio.get("success", 0)
        off_tot = self._offspring_ratio.get("total", 0)
        off_rte = f"{off_suc / off_tot if off_tot > 0 else 0.0:.2%}"
        unique_inds = len(self.total_population)
        total_inds = len(self)
        unique_rte = f"{unique_inds / total_inds if total_inds > 0 else 0.0:.2%}"
        final_log += (
            f"\n  Total islands          : {len(self.islands)}"
            f"\n  Unique individuals     : {unique_inds}/{total_inds} ({unique_rte})"
            f"\n  Crossover success rate : {sum_crs_suc}/{sum_crs_tot} ({sum_crs_rte})"
            f"\n  Mutation success rate  : {sum_mut_suc}/{sum_mut_tot} ({sum_mut_rte})"
            f"\n  Offspring success rate : {off_suc}/{off_tot} ({off_rte})"
        )
        self.context.logger.debug(final_log)

    def finalize(self) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        self._deduplicate_population()
        self._aggregate_stats_from_islands()
        self._log_operators(name="crossover", ratios=self._crossover_ratio, ops=self.context.crossovers)
        self._log_operators(name="mutation", ratios=self._mutation_ratio, ops=self.context.mutations)
        self._log_aggregate_stats()
        self._log_unique_events()

    def _log_unique_events(self) -> None:
        """Count unique event lists."""
        unique_genes = Counter()
        for schedule in self.total_population:
            for team in schedule.all_teams():
                unique_gene = tuple(sorted(team.events))
                unique_genes[unique_gene] += 1
        for gene, count in sorted(unique_genes.items(), key=lambda x: x[1], reverse=True):
            self.context.logger.debug("Event: %s, Count: %d", gene, count)
