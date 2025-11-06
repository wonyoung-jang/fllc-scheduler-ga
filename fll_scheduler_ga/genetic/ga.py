"""Genetic algorithm for FLL Scheduler GA."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..io.observers import LoggingObserver, TqdmObserver
from .island import Island

if TYPE_CHECKING:
    from ..config.schemas import GaParameters, TournamentConfig
    from ..data_model.schedule import Schedule
    from ..io.observers import GaObserver
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from .ga_context import GaContext

logger = getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    rng: np.random.Generator
    observers: tuple[GaObserver]
    seed_file: Path
    save_front_only: bool

    curr_gen: int = 0
    fitness_history: np.ndarray = None
    total_population: list[Schedule] = field(default_factory=list, repr=False)
    islands: list[Island] = field(default_factory=list, repr=False)
    ga_params: GaParameters = None

    generations_array: np.ndarray = None
    migrate_generations: np.ndarray = None

    offspring_ratio: Counter = field(default_factory=Counter, repr=False)
    crossover_ratio: dict[str, Counter] = None
    mutation_ratio: dict[str, Counter] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.ga_params = self.context.app_config.ga_params
        self.generations_array = np.arange(1, self.ga_params.generations + 1)
        self.migrate_generations: np.ndarray = np.zeros(self.ga_params.generations + 1, dtype=int)
        if self.ga_params.num_islands > 1 and self.ga_params.migration_size > 0:
            self.migrate_generations[:: self.ga_params.migration_interval] = 1

        self.fitness_history = np.full(
            (self.ga_params.generations, len(self.context.evaluator.objectives)),
            fill_value=-1,
            dtype=float,
        )

        trackers = ("success", "total")
        self.crossover_ratio = {tr: Counter({str(c): 0 for c in self.context.crossovers}) for tr in trackers}
        self.mutation_ratio = {tr: Counter({str(m): 0 for m in self.context.mutations}) for tr in trackers}

        self.islands.extend(
            Island(
                identity=i,
                context=self.context,
                offspring_ratio=self.offspring_ratio.copy(),
                crossover_ratio=self.crossover_ratio.copy(),
                mutation_ratio=self.mutation_ratio.copy(),
            )
            for i in range(self.ga_params.num_islands)
        )

    @classmethod
    def build(cls, context: GaContext) -> GA:
        """Build and return a GA instance with the provided configuration."""
        return cls(
            context=context,
            rng=context.app_config.rng,
            observers=(TqdmObserver(), LoggingObserver()),
            seed_file=Path(context.app_config.runtime.seed_file),
            save_front_only=context.app_config.exports.front_only,
        )

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        start_time = time()
        self._notify_on_start(self.ga_params.generations)
        try:
            seed_pop = GALoad(self).load()
            self.initialize_population(seed_pop)
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
            GAFinalizer(self).finalize(start_time)
            GASave(self).save()
            self._notify_on_finish(self.total_population, self.pareto_front())

    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        if not self.total_population:
            return [p for i in self.islands for p in i.pareto_front()]
        return [p for p in self.total_population if p.rank == 0]

    def get_this_gen_fitness(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        island_fitnesses = np.asarray([i.get_last_gen_fitness() for i in self.islands], dtype=float)
        return island_fitnesses.mean(axis=0)

    def get_last_gen_fitness(self) -> np.ndarray:
        """Get the fitness of the last generation."""
        return self.fitness_history[self.curr_gen - 1] if self.curr_gen > 0 else ()

    def update_fitness_history(self) -> None:
        """Update the fitness history with the current generation's fitness."""
        self.fitness_history[self.curr_gen] = self.get_this_gen_fitness()

    def initialize_population(self, seed_pop: list[Schedule] | None) -> None:
        """Initialize the population for each island."""
        if seed_pop is not None:
            self.rng.shuffle(seed_pop)
            n = self.ga_params.num_islands
            for idx, schedule in enumerate(seed_pop):
                self.islands[idx % n].add_to_population(schedule)

        logger.debug("Initializing %d islands...", self.ga_params.num_islands)

        for island in self.islands:
            island.initialize()
            island.evaluate_pop()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.generations_array:
            if self.migrate_generations[gen]:
                self.migrate()

            self.run_generation()
            self.update_fitness_history()
            self.curr_gen += 1
            self._notify_on_generation_end(
                generation=gen,
                num_generations=self.ga_params.generations,
                best_fitness=self.get_last_gen_fitness(),
                pop_size=len(self),
            )

    def run_generation(self) -> None:
        """Run a single epoch of the genetic algorithm."""
        for island in self.islands:
            island.handle_underpopulation()
            island.evolve()
            island.select_next_generation()
            island.update_fitness_history()
            island.curr_gen += 1

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        n = len(self.islands)
        for i, receiving_island in enumerate(self.islands):
            giving_island = self.islands[(i + 1) % n]
            migrants = giving_island.give_migrants()
            receiving_island.receive_migrants(migrants)

    def _notify_on_start(self, num_generations: int) -> None:
        """Notify observers when the genetic algorithm run starts."""
        for obs in self.observers:
            obs.on_start(num_generations)

    def _notify_on_generation_end(
        self, generation: int, num_generations: int, best_fitness: np.ndarray, pop_size: int
    ) -> None:
        """Notify observers at the end of a generation."""
        for obs in self.observers:
            obs.on_generation_end(generation, num_generations, best_fitness, pop_size)

    def _notify_on_finish(self, pop: list[Schedule], pareto_front: list[Schedule]) -> None:
        """Notify observers when the genetic algorithm run is finished."""
        for obs in self.observers:
            obs.on_finish(pop, pareto_front)


@dataclass(slots=True)
class GAFinalizer:
    """Finalizer for GA instances."""

    ga: GA

    def finalize(self, start_time: float) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        ga = self.ga
        ctx = ga.context

        self._deduplicate_population()
        self._aggregate_stats_from_islands()
        self._log_operators(name="crossover", ratios=ga.crossover_ratio, ops=ctx.crossovers)
        self._log_operators(name="mutation", ratios=ga.mutation_ratio, ops=ctx.mutations)
        self._log_aggregate_stats()
        for island in ga.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time() - start_time)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        ga = self.ga
        ctx = ga.context

        unique_pop = [ind for island in ga.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in ga.islands for s in island.selected])
        schedule_fitness, team_fitnesses = ctx.evaluator.evaluate_population(pop_array)
        fronts = ctx.nsga3.non_dominated_sort(schedule_fitness, len(pop_array))

        selected = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                idx: int
                sch = unique_pop[idx]
                sch.fitness = schedule_fitness[idx]
                sch.team_fitnesses = team_fitnesses[idx]
                sch.rank = rank
                selected[hash(sch)] = sch

        ga.total_population = sorted(selected.values(), key=lambda s: (s.rank, -s.fitness.sum()))

    def _aggregate_stats_from_islands(self) -> None:
        """Aggregate statistics from all islands."""
        ga = self.ga
        for island in ga.islands:
            for tracker, crossovers in island.crossover_ratio.items():
                ga.crossover_ratio[tracker].update(crossovers)
            for tracker, mutations in island.mutation_ratio.items():
                ga.mutation_ratio[tracker].update(mutations)
            for tracker, count in island.offspring_ratio.items():
                ga.offspring_ratio[tracker] += count

    @staticmethod
    def _log_operators(name: str, ratios: dict[str, Counter], ops: tuple[Crossover | Mutation]) -> None:
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
        ga = self.ga
        final_log = f"{'=' * 20}\nFinal statistics"
        crs_suc = sum(ga.crossover_ratio.get("success", {}).values())
        crs_tot = sum(ga.crossover_ratio.get("total", {}).values())
        crs_rte = f"{crs_suc / crs_tot if crs_tot > 0 else 0.0:.2%}"
        mut_suc = sum(ga.mutation_ratio.get("success", {}).values())
        mut_tot = sum(ga.mutation_ratio.get("total", {}).values())
        mut_rte = f"{mut_suc / mut_tot if mut_tot > 0 else 0.0:.2%}"
        off_suc = ga.offspring_ratio.get("success", 0)
        off_tot = ga.offspring_ratio.get("total", 0)
        off_rte = f"{off_suc / off_tot if off_tot > 0 else 0.0:.2%}"
        unique_inds = len(ga.total_population)
        total_inds = len(ga)
        unique_rte = f"{unique_inds / total_inds if total_inds > 0 else 0.0:.2%}"
        final_log += (
            f"\n  Total islands          : {len(ga.islands)}"
            f"\n  Unique individuals     : {unique_inds}/{total_inds} ({unique_rte})"
            f"\n  Crossover success rate : {crs_suc}/{crs_tot} ({crs_rte})"
            f"\n  Mutation success rate  : {mut_suc}/{mut_tot} ({mut_rte})"
            f"\n  Offspring success rate : {off_suc}/{off_tot} ({off_rte})"
        )
        logger.debug(final_log)


@dataclass(slots=True)
class GALoad:
    """Loader for GA instances from seed files."""

    ga: GA

    def load(self) -> list[Schedule] | None:
        """Load and integrate a population from a seed file."""
        ga = self.ga
        logger.debug("Loading seed population from: %s", ga.seed_file)
        seed_data: dict[str, Any] = {}
        try:
            with ga.seed_file.open("rb") as f:
                seed_data = pickle.load(f)
        except (OSError, pickle.PicklingError):
            logger.exception("Could not load or parse seed file. Starting with a fresh population.")
        except EOFError:
            logger.debug("Pickle file is empty")

        seed_config: TournamentConfig | None = seed_data.get("config")
        if seed_config is None:
            logger.warning("Seed population is missing config. Using current...")
            return None

        if ga.context.app_config.tournament != seed_config:
            logger.warning("Seed population does not match current config. Using current...")
            return None

        population = seed_data.get("population")
        if population is None:
            logger.warning("Seed population is missing. Using current...")
            return None

        return population


@dataclass(slots=True)
class GASave:
    """Saver for GA instances to seed files."""

    ga: GA

    def save(self) -> None:
        """Save the final population to a file to be used as a seed for a future run."""
        ga = self.ga
        pop = ga.pareto_front() if ga.save_front_only else ga.total_population

        if not pop:
            logger.warning("No population to save to seed file.")
            return

        data_to_cache = {
            "population": pop,
            "config": ga.context.app_config.tournament,
        }

        path = ga.seed_file
        logger.debug("Saving final population of size %d to seed file: %s", len(pop), path)

        try:
            with path.open("wb") as f:
                pickle.dump(data_to_cache, f)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Error saving population to seed file: %s", path)
