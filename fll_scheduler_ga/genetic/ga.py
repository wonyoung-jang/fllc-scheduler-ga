"""Genetic algorithm for FLL Scheduler GA."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.constants import DATA_MODEL_VERSION, SeedIslandStrategy, SeedPopSort
from ..io.observers import LoggingObserver, TqdmObserver
from ..io.seed_ga import GALoad, GASave, GASeedData
from .ga_generation import GaGeneration
from .island import Island
from .stagnation import FitnessHistory, OperatorStats

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.schemas import GaParameters, GeneticModel, ImportModel
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
    genetic_model: GeneticModel
    rng: np.random.Generator
    observers: tuple[GaObserver]
    seed_file: Path
    save_front_only: bool

    generation: GaGeneration = None
    total_population: list[Schedule] = field(default_factory=list, repr=False)
    islands: list[Island] = field(default_factory=list, repr=False)

    fitness_history: FitnessHistory = None
    operator_stats: OperatorStats = None
    generations_array: np.ndarray = None
    migrate_generations: np.ndarray = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.generation = GaGeneration(curr=0)

        self.generations_array = np.arange(1, self.genetic_model.parameters.generations + 1)
        self.migrate_generations = np.zeros(self.genetic_model.parameters.generations + 1, dtype=int)
        if self.genetic_model.parameters.num_islands > 1 and self.genetic_model.parameters.migration_size > 0:
            self.migrate_generations[:: self.genetic_model.parameters.migration_interval] = 1

        self._init_fitness_history()
        self._init_operator_stats()

        self.islands.extend(
            Island(
                identity=i,
                generation=self.generation,
                context=self.context,
                genetic_model=self.genetic_model,
                rng=self.rng,
                operator_stats=self.operator_stats,
                fitness_history=self.fitness_history.copy(),
                builder=self.context.builder,
            )
            for i in range(self.genetic_model.parameters.num_islands)
        )

    @classmethod
    def build(cls, context: GaContext) -> GA:
        """Build and return a GA instance with the provided configuration."""
        _app_config = context.app_config
        return cls(
            context=context,
            genetic_model=_app_config.genetic,
            rng=_app_config.rng,
            observers=(TqdmObserver(), LoggingObserver()),
            seed_file=Path(_app_config.runtime.seed_file),
            save_front_only=_app_config.exports.front_only,
        )

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    def _init_fitness_history(self) -> None:
        """Initialize the fitness history for the GA."""
        n_gen = self.genetic_model.parameters.generations
        n_obj = len(self.context.evaluator.objectives)
        self.fitness_history = FitnessHistory(
            generation=self.generation,
            current=np.zeros((1, n_obj), dtype=float),
            history=np.full((n_gen, n_obj), fill_value=-1, dtype=float),
        )

    def _init_operator_stats(self) -> None:
        """Initialize the operator statistics for the GA."""
        trackers = ("success", "total")
        crossover_counters = {str(c): 0 for c in self.context.crossovers}
        mutation_counters = {str(m): 0 for m in self.context.mutations}
        self.operator_stats = OperatorStats(
            offspring=Counter(),
            crossover={tr: Counter(crossover_counters) for tr in trackers},
            mutation={tr: Counter(mutation_counters) for tr in trackers},
        )

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        seed_file = self.seed_file
        config = self.context.app_config.tournament
        try:
            start_time = time()
            self._notify_on_start(self.genetic_model.parameters.generations)
            seed_data = GALoad(
                seed_file=seed_file,
                config=config,
                evaluator=self.context.evaluator,
            ).load()
            if seed_data is not None:
                self.seed_population(seed_data)
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                logger.critical("No valid schedule meeting all hard constraints was found.")
                return False
            self.run_epochs()
        except Exception:
            logger.exception("An error occurred during the genetic algorithm run.")
            self.fitness_history.current = self.aggregate_island_fitness()
            self.fitness_history.update_fitness_history()
        except KeyboardInterrupt:
            logger.debug("Genetic algorithm run interrupted by user. Saving...")
            self.fitness_history.current = self.aggregate_island_fitness()
            self.fitness_history.update_fitness_history()
        finally:
            GAFinalizer(self).finalize(start_time)
            seed_ga_data = GASeedData(
                version=DATA_MODEL_VERSION,
                config=config,
                population=self.pareto_front() if self.save_front_only else self.total_population,
            )
            GASave(
                seed_file=seed_file,
                data=seed_ga_data,
            ).save()
            self._notify_on_finish(self.total_population, self.pareto_front())

    def pareto_front(self) -> list[Schedule]:
        """Get the Pareto front for each island in the population."""
        return [p for p in self.total_population if p.rank == 0]

    def aggregate_island_fitness(self) -> np.ndarray:
        """Calculate the average fitness of the current generation."""
        island_fitnesses = np.asarray([i.fitness_history.get_last_gen_fitness() for i in self.islands], dtype=float)
        return island_fitnesses.mean(axis=0)

    def seed_population(self, seed_data: GASeedData) -> None:
        """Seed the population for each island."""
        seeder = GASeeder(
            imports=self.context.app_config.imports,
            ga_params=self.genetic_model.parameters,
            seed_pop=seed_data.population,
            rng=self.rng,
        )
        if not seeder.is_valid():
            return

        island_to_seed_idx = seeder.get_island_seeds()
        for i, seed_indices in island_to_seed_idx.items():
            island = self.islands[i]
            for idx in seed_indices:
                if island.add_to_population(seed_data.population[idx]):
                    island.population.add_schedule(seed_data.population[idx].schedule)

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        logger.debug("Initializing %d islands...", self.genetic_model.parameters.num_islands)
        for island in self.islands:
            island.initialize()

    def run_epochs(self) -> None:
        """Perform main evolution loop: generations and migrations."""
        for gen in self.generations_array:
            if self.migrate_generations[gen]:
                self.migrate()

            # Run the generations
            for island in self.islands:
                island.run_epoch()

            self.fitness_history.current = self.aggregate_island_fitness()
            self.fitness_history.update_fitness_history()

            self.generation.increment()

            self._notify_on_generation_end(
                generation=gen,
                num_generations=self.genetic_model.parameters.generations,
                best_fitness=self.fitness_history.get_last_gen_fitness(),
                pop_size=len(self),
            )

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
class GASeeder:
    """Seeding strategies for GA instances."""

    imports: ImportModel
    ga_params: GaParameters
    seed_pop: list[Schedule] | None
    rng: np.random.Generator

    def is_valid(self) -> bool:
        """Check if seeding is valid based on the provided seed population."""
        if not self.seed_pop or self.seed_pop is None:
            logger.debug("No seed population provided. Starting with a fresh population.")
            return False
        logger.debug("Seeding population with %d individuals from seed file.", len(self.seed_pop))
        logger.debug(
            "Seed pop sort: %s | Seed island strategy: %s",
            self.imports.seed_pop_sort,
            self.imports.seed_island_strategy,
        )
        return True

    def get_island_seeds(self) -> dict[int, list[int]]:
        """Get seed indices for each island."""
        params = {
            "seed_indices": self._iter_seeds(),
            "n_islands": self.ga_params.num_islands,
            "n_pop": self.ga_params.population_size,
        }
        seed_fn = {
            SeedIslandStrategy.DISTRIBUTED: self._seed_distributed,
            SeedIslandStrategy.CONCENTRATED: self._seed_concentrated,
        }.get(self.imports.seed_island_strategy, self._seed_distributed)
        return seed_fn(**params)

    def _iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        iter_fn = {
            SeedPopSort.RANDOM: self.rng.permutation,
            SeedPopSort.BEST: np.arange,
        }.get(self.imports.seed_pop_sort, self.rng.permutation)
        yield from iter_fn(len(self.seed_pop))

    def _seed_distributed(self, **params: dict[str, Any]) -> dict[int, list[int]]:
        """Get seed indices for distributed seeding strategy."""
        island_to_seed_idx: dict[int, list[int]] = defaultdict(list)
        for idx in params["seed_indices"]:
            island_to_seed_idx[idx % params["n_islands"]].append(idx)
        return island_to_seed_idx

    def _seed_concentrated(self, **params: dict[str, Any]) -> dict[int, list[int]]:
        """Get seed indices for concentrated seeding strategy."""
        island_to_seed_idx: dict[int, list[int]] = defaultdict(list)
        for i in range(params["n_islands"]):
            while len(island_to_seed_idx[i]) < params["n_pop"]:
                if (idx := next(params["seed_indices"], None)) is None:
                    break
                island_to_seed_idx[i].append(idx)
        return island_to_seed_idx


@dataclass(slots=True)
class GAFinalizer:
    """Finalizer for GA instances."""

    ga: GA

    def finalize(self, start_time: float) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        ga = self.ga
        ctx = ga.context

        self._deduplicate_population()
        self._log_operators(name="crossover", ratios=ga.operator_stats.crossover, ops=ctx.crossovers)
        self._log_operators(name="mutation", ratios=ga.operator_stats.mutation, ops=ctx.mutations)
        self._log_aggregate_stats()
        for island in ga.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.fitness_history.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time() - start_time)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        ga = self.ga
        ctx = ga.context

        unique_pop = [ind for island in ga.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in ga.islands for s in island.selected])
        schedule_fitness, team_fitnesses = ctx.evaluator.evaluate_population(pop_array)
        _, flat, ranks = ctx.nsga3.select(schedule_fitness, len(unique_pop))

        selected = {}
        for rank, idx in zip(ranks, flat, strict=True):
            idx: int
            sch = unique_pop[idx]
            sch.fitness = schedule_fitness[idx]
            sch.team_fitnesses = team_fitnesses[idx]
            sch.rank = rank
            selected[hash(sch)] = sch

        ga.total_population = sorted(selected.values(), key=lambda s: (s.rank, -s.fitness.sum()))

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
        crs_suc = sum(ga.operator_stats.crossover.get("success", {}).values())
        crs_tot = sum(ga.operator_stats.crossover.get("total", {}).values())
        crs_rte = f"{crs_suc / crs_tot if crs_tot > 0 else 0.0:.2%}"
        mut_suc = sum(ga.operator_stats.mutation.get("success", {}).values())
        mut_tot = sum(ga.operator_stats.mutation.get("total", {}).values())
        mut_rte = f"{mut_suc / mut_tot if mut_tot > 0 else 0.0:.2%}"
        off_suc = ga.operator_stats.offspring.get("success", 0)
        off_tot = ga.operator_stats.offspring.get("total", 0)
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
