"""Genetic algorithm for FLL Scheduler GA."""

import pickle
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from random import Random
from time import time
from typing import TYPE_CHECKING, Any

from ..config.constants import RANDOM_SEED_RANGE
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

logger = getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    rng: Random
    observers: tuple[GaObserver]

    fitness_history: list[tuple] = field(default_factory=list, repr=False)
    total_population: Population = field(default_factory=list, repr=False)
    islands: list[Island] = field(default_factory=list, repr=False)
    ga_params: GaParameters = field(default=None, repr=False)

    _seed_file: Path | None = field(default=None, repr=False)
    _offspring_ratio: Counter = field(default_factory=Counter, repr=False)
    _crossover_ratio: dict[str, Counter] = field(default=None, repr=False)
    _mutation_ratio: dict[str, Counter] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.ga_params = self.context.app_config.ga_params
        seeder = Random(self.rng.randint(*RANDOM_SEED_RANGE))
        builder = ScheduleBuilder(
            team_factory=self.context.team_factory,
            event_factory=self.context.event_factory,
            config=self.context.app_config.tournament,
            rng=Random(seeder.randint(*RANDOM_SEED_RANGE)),
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

        trackers = ("success", "total")
        self._crossover_ratio = {trckr: Counter({str(c): 0 for c in self.context.crossovers}) for trckr in trackers}
        self._mutation_ratio = {trckr: Counter({str(m): 0 for m in self.context.mutations}) for trckr in trackers}

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
            sum(s) / self.ga_params.num_islands
            for s in zip(
                *(i.get_last_gen_fitness() for i in self.islands),
                strict=True,
            )
        )

    def _get_last_gen_fitness(self) -> tuple[float, ...]:
        """Get the fitness of the last generation."""
        return self.fitness_history[-1] if self.fitness_history else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history.append(self._get_this_gen_fitness())

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time()
        self._notify_on_start(self.ga_params.generations)

        try:
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
        except Exception:
            logger.exception("An error occurred during the genetic algorithm run.")
            self.update_fitness_history()
        except KeyboardInterrupt:
            logger.debug("Genetic algorithm run interrupted by user. Saving...")
            self.update_fitness_history()
        finally:
            self.finalize(start_time)
            self._notify_on_finish(self.total_population, self.pareto_front())

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        seed_path = self._seed_file
        if seed_path and seed_path.exists() and (seed_pop := self.retrieve_seed_population()):
            seed_pop.sort(key=lambda s: (s.rank, -sum(s.fitness)))
            n = self.ga_params.num_islands
            for idx, schedule in enumerate(seed_pop):
                self.islands[idx % n].add_to_population(schedule)

        logger.debug("Initializing %d islands...", self.ga_params.num_islands)
        for i in range(self.ga_params.num_islands):
            self.islands[i].initialize()

    def retrieve_seed_population(self) -> Population | None:
        """Load and integrate a population from a seed file."""
        logger.debug("Loading seed population from: %s", self._seed_file)
        seed_data: dict[str, Any] = {}
        try:
            with self._seed_file.open("rb") as f:
                seed_data = pickle.load(f)
        except (OSError, pickle.PicklingError):
            logger.exception("Could not load or parse seed file. Starting with a fresh population.")
        except EOFError:
            logger.debug("Pickle file is empty")

        seed_config: TournamentConfig | None = seed_data.get("config")
        if not seed_config:
            logger.warning("Seed population is missing config. Using current...")
            return None

        num_teams_changed = self.context.app_config.tournament.num_teams != seed_config.num_teams
        rounds_changed = self.context.app_config.tournament.rounds != seed_config.rounds
        if num_teams_changed or rounds_changed:
            logger.warning("Seed population does not match current config. Using current...")
            return None

        population = seed_data.get("population")
        if not population:
            logger.warning("Seed population is missing. Using current...")
            return None

        return population

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        num_generations = self.ga_params.generations
        for generation in range(1, num_generations + 1):
            self.migrate(generation)
            self.run_single_epoch(generation - 1)
            self.update_fitness_history()
            self._notify_on_generation_end(
                generation=generation,
                num_generations=num_generations,
                best_fitness=self._get_last_gen_fitness(),
            )

    def run_single_epoch(self, generation: int) -> None:
        """Run a single epoch of the genetic algorithm."""
        for island in self.islands:
            island.evolve(generation)
            island.update_fitness_history()

    def migrate(self, generation: int) -> None:
        """Migrate the best individuals between islands using a random ring topology."""
        num_islands = self.ga_params.num_islands
        if not (
            self.ga_params.migration_size > 0
            and generation % self.ga_params.migration_interval == 0
            and num_islands > 1
        ):
            return

        islands = self.islands
        o = self.rng.randrange(0, num_islands)  # Random offset
        for i, island in enumerate(islands):
            j = (i + o) % num_islands
            island.receive_migrants(islands[j].get_migrants())

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_on_generation_end(
        self,
        generation: int,
        num_generations: int,
        best_fitness: tuple[float, ...],
    ) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(generation, num_generations, best_fitness)

    def _notify_on_finish(self, pop: Population, pareto_front: Population) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, pareto_front)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        unique_pop = list({ind for island in self.islands for ind in island.finalize_island()})
        selected = self.context.nsga3.select(unique_pop, population_size=len(unique_pop))
        self.total_population = sorted(selected.values(), key=lambda s: (s.rank, -sum(s.fitness)))

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
        if not (op_strings := [f"{op!s}" for op in ops]):
            return

        log = f"{name.capitalize()} statistics:"
        max_len = max(len(s) for s in op_strings) + 1
        for op in op_strings:
            success = ratios.get("success", {}).get(op, 0)
            total = ratios.get("total", {}).get(op, 0)
            rate = success / total if total > 0 else 0.0
            log += f"\n  {op:<{max_len}}: {success}/{total} ({rate:.2%})"
        logger.debug(log)

    def _log_aggregate_stats(self) -> None:
        """Log aggregate statistics across all islands."""
        final_log = f"{'=' * 20}\nFinal statistics"
        crs_suc = sum(self._crossover_ratio.get("success", {}).values())
        crs_tot = sum(self._crossover_ratio.get("total", {}).values())
        crs_rte = f"{crs_suc / crs_tot if crs_tot > 0 else 0.0:.2%}"
        mut_suc = sum(self._mutation_ratio.get("success", {}).values())
        mut_tot = sum(self._mutation_ratio.get("total", {}).values())
        mut_rte = f"{mut_suc / mut_tot if mut_tot > 0 else 0.0:.2%}"
        off_suc = self._offspring_ratio.get("success", 0)
        off_tot = self._offspring_ratio.get("total", 0)
        off_rte = f"{off_suc / off_tot if off_tot > 0 else 0.0:.2%}"
        unique_inds = len(self.total_population)
        total_inds = len(self)
        unique_rte = f"{unique_inds / total_inds if total_inds > 0 else 0.0:.2%}"
        final_log += (
            f"\n  Total islands          : {len(self.islands)}"
            f"\n  Unique individuals     : {unique_inds}/{total_inds} ({unique_rte})"
            f"\n  Crossover success rate : {crs_suc}/{crs_tot} ({crs_rte})"
            f"\n  Mutation success rate  : {mut_suc}/{mut_tot} ({mut_rte})"
            f"\n  Offspring success rate : {off_suc}/{off_tot} ({off_rte})"
        )
        logger.debug(final_log)

    def _log_unique_events(self) -> None:
        """Count unique event lists."""
        unique_genes = Counter()
        for schedule in self.total_population:
            for team in schedule.all_teams():
                unique_genes[tuple(sorted(team.events))] += 1
        for gene, count in unique_genes.most_common(n=10):
            logger.debug("Event: %s, Count: %d", gene, count)

    def finalize(self, start_time: float) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        self._deduplicate_population()
        self._log_unique_events()
        self._aggregate_stats_from_islands()
        self._log_operators(name="crossover", ratios=self._crossover_ratio, ops=self.context.crossovers)
        self._log_operators(name="mutation", ratios=self._mutation_ratio, ops=self.context.mutations)
        self._log_aggregate_stats()
        logger.debug("Fitness caches: %s", self.context.evaluator.cache_info())
        logger.debug("GaParameter Values: %s", str(self.ga_params))
        for island in self.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time() - start_time)
