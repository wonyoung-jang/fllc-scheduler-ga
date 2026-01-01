"""Genetic algorithm for FLL Scheduler GA."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import SeedIslandStrategy, SeedPopSort
from ..io.seed_ga import (
    ConcentratedSeedingStrategy,
    DistributedSeedingStrategy,
    GALoad,
    GASave,
    GASeedData,
    SeedingStrategy,
)
from .island import Island
from .population import SchedulePopulation
from .stagnation import FitnessHistory, OperatorStats, StagnationHandler

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterator
    from pathlib import Path

    from ..config.pydantic_schemas import GaParameterModel, GeneticModel, ImportModel
    from ..data_model.schedule import Schedule
    from ..io.observers import GaObserver
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from .ga_context import GaContext
    from .ga_generation import GaGeneration

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GA:
    """Genetic algorithm for the FLL Scheduler GA."""

    context: GaContext
    genetic_model: GeneticModel
    rng: np.random.Generator
    observers: tuple[GaObserver, ...]
    seed_file: Path
    save_front_only: bool
    generation: GaGeneration
    operator_stats: OperatorStats
    fitness_history: FitnessHistory
    generations_array: np.ndarray
    migrate_generations: np.ndarray

    total_population: list[Schedule] = field(default_factory=list)
    islands: list[Island] = field(default_factory=list)

    _n_islands: int = field(init=False)
    _n_generations: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        params = self.genetic_model.parameters
        self._n_islands = params.num_islands
        self._n_generations = params.generations

        self.initialize_islands()

    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return sum(len(i) for i in self.islands)

    def initialize_islands(self) -> None:
        """Initialize all islands in the GA."""
        n_islands = self._n_islands
        for i in range(n_islands):
            island = Island(
                identity=i,
                generation=self.generation,
                context=self.context,
                genetic_model=self.genetic_model,
                rng=self.rng,
                operator_stats=self.operator_stats,
                fitness_history=self.fitness_history.copy(),
                builder=self.context.builder,
                population=SchedulePopulation(
                    ranks=np.empty((0,), dtype=int),
                ),
            )
            island.stagnation = StagnationHandler(
                rng=self.rng,
                generation=self.generation,
                fitness_history=island.fitness_history,
                model=self.genetic_model.stagnation,
            )
            self.islands.append(island)

    def run(self) -> None:
        """Run the genetic algorithm and return the best schedule found."""
        seed_file = self.seed_file
        config = self.context.get_tournament_config()
        try:
            start_time = time.time()
            self._notify_on_start(self._n_generations)
            seed_data = GALoad(
                seed_file=seed_file,
                config=config,
            ).load()
            if seed_data is not None:
                self.seed_population(seed_data)
            self.initialize_population()
            if not any(i.selected for i in self.islands):
                logger.critical("No valid schedule meeting all hard constraints was found.")
                return
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
        seed_strategy_map = {
            SeedIslandStrategy.DISTRIBUTED: DistributedSeedingStrategy,
            SeedIslandStrategy.CONCENTRATED: ConcentratedSeedingStrategy,
        }
        seed_strategy = seed_strategy_map.get(self.context.get_seed_island_strategy(), DistributedSeedingStrategy)
        seeder = GASeeder(
            strategy=seed_strategy(),
            imports=self.context.get_imports_model(),
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
                    island.population.add(seed_data.population[idx].schedule)

    def initialize_population(self) -> None:
        """Initialize the population for each island."""
        logger.debug("Initializing %d islands...", self._n_islands)
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
                num_generations=self._n_generations,
                best_fitness=self.fitness_history.get_last_gen_fitness(),
                pop_size=len(self),
            )

    def migrate(self) -> None:
        """Migrate the best individuals between islands using a ring topology."""
        n = len(self.islands)
        for i, dest in enumerate(self.islands):
            src = self.islands[(i + 1) % n]
            migrants = src.give_migrants()
            dest.receive_migrants(migrants)

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

    strategy: SeedingStrategy
    imports: ImportModel
    ga_params: GaParameterModel
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
        seed_indices = self._iter_seeds()
        n_islands = self.ga_params.num_islands
        n_pop = self.ga_params.population_size
        return self.strategy.get_indices(seed_indices, n_islands, n_pop)

    def _iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        iter_fn = {
            SeedPopSort.RANDOM: self.rng.permutation,
            SeedPopSort.BEST: np.arange,
        }.get(self.imports.seed_pop_sort, self.rng.permutation)
        if isinstance(self.seed_pop, list):
            yield from iter_fn(len(self.seed_pop))


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
        self._log_aggregate_stats(ga.operator_stats)
        for island in ga.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.fitness_history.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time.time() - start_time)

    def _deduplicate_population(self) -> None:
        """Remove duplicate individuals from the population."""
        ga = self.ga
        ctx = ga.context

        unique_pop = [ind for island in ga.islands for ind in island.selected]
        pop_array = np.asarray([s.schedule for island in ga.islands for s in island.selected])
        schedule_fitness, team_fitnesses = ctx.evaluate(pop_array)
        _, flat, ranks = ctx.select_nsga3(schedule_fitness, len(unique_pop))

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
    def _log_operators(name: str, ratios: dict[str, Counter], ops: tuple[Crossover | Mutation, ...]) -> None:
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

    def _log_aggregate_stats(self, operator_stats: OperatorStats) -> None:
        """Log aggregate statistics across all islands."""
        ga = self.ga
        final_log = f"{'=' * 20}\nFinal statistics"
        crs_suc, crs_tot, crs_rte = operator_stats.get_crossover_stats()
        mut_suc, mut_tot, mut_rte = operator_stats.get_mutation_stats()
        off_suc, off_tot, off_rte = operator_stats.get_offspring_stats()
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
