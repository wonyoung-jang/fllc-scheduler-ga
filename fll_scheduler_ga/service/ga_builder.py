"""Builder for the GA instance based on the provided configuration and context."""

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.monitoring import LoggingObserver, RichObserver
from fll_scheduler_ga.constants import SeedIslandStrategy, SeedPopSort
from fll_scheduler_ga.genetic.ga import GA, FitnessHistory, Island, OperatorStats, StagnationHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.adapter.schema import AppConfig, GaParameterModel
    from fll_scheduler_ga.domain.model import GaObserver
    from fll_scheduler_ga.genetic.context import GaContext


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GASeeder:
    """Seeding strategies for GA instances."""

    rng: np.random.Generator
    popsize: int
    island_strategy: SeedIslandStrategy
    popsort: SeedPopSort
    n_islands: int
    n_pop: int

    def distributed_seeding(self) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        result: dict[int, list[int]] = defaultdict(list)
        for idx in self.iter_seeds():
            result[idx % self.n_islands].append(idx)
        return result

    def concentrated_seeding(self) -> dict[int, list[int]]:
        """Get the seed indices for each island."""
        inds = self.iter_seeds()
        result: dict[int, list[int]] = defaultdict(list)
        for i in range(self.n_islands):
            while len(result[i]) < self.n_pop:
                if (idx := next(inds, None)) is None:
                    break
                result[i].append(idx)
        return result

    def get_island_seed_map(self) -> dict[int, list[int]]:
        """Get the mapping of islands to seed indices based on the seeding strategy."""
        if not self.popsize:
            logger.debug("No seed population provided. Starting with a fresh population.")
            return {}
        logger.debug(
            "Seeding population file.\nSeed pop size: %s\nSeed pop sort: %s | Seed island strategy: %s",
            self.popsize,
            self.popsort,
            self.island_strategy,
        )
        match self.island_strategy:
            case SeedIslandStrategy.CONCENTRATED:
                return self.concentrated_seeding()
            case SeedIslandStrategy.DISTRIBUTED:
                return self.distributed_seeding()

    def iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        match self.popsort:
            case SeedPopSort.BEST:
                yield from np.arange(self.popsize)
            case SeedPopSort.RANDOM:
                yield from self.rng.permutation(self.popsize)


def _get_tracker(operators: tuple) -> dict[str, Counter]:
    inner = {str(o): 0 for o in operators}
    return {"success": Counter(inner), "total": Counter(inner)}


def _get_migration_generations(gp: GaParameterModel) -> np.ndarray:
    """Calculate the generations at which migration should occur."""
    gen = np.zeros(gp.generations + 1, dtype=int)
    if gp.num_islands > 1 and gp.migration_size > 0:
        gen[:: gp.migration_interval] = 1
    return gen


def _get_observers(progress: Progress | None, task_id: TaskID | None) -> tuple[GaObserver, ...]:
    """Get the observers for the GA."""
    observers: list[GaObserver] = [LoggingObserver()]
    if progress and task_id is not None:
        observers.append(RichObserver(progress, task_id))
    return tuple(observers)


@dataclass(slots=True)
class GaBuilder:
    """Builder for the GA instance based on the provided configuration and context."""

    progress: Progress | None
    task_id: TaskID | None
    cfg: AppConfig
    ctx: GaContext
    seed_pop: list

    def build(self) -> GA:
        """Build the GA instance."""
        seeder = GASeeder(
            self.cfg.rng,
            len(self.seed_pop),
            self.cfg.io.imports.seed_island_strategy,
            self.cfg.io.imports.seed_pop_sort,
            self.cfg.genetic.parameters.num_islands,
            self.cfg.genetic.parameters.population_size,
        )
        operator_stats = OperatorStats(Counter(), _get_tracker(self.ctx.crossovers), _get_tracker(self.ctx.mutations))
        fitness_history = FitnessHistory(
            generation=0,
            current=np.zeros((1, self.ctx.evaluator.n_objectives), dtype=float),
            history=np.full(
                (self.cfg.genetic.parameters.generations, self.ctx.evaluator.n_objectives), fill_value=-1, dtype=float
            ),
        )
        stagnation = StagnationHandler(
            enabled=self.cfg.genetic.stagnation.enable,
            threshold=self.cfg.genetic.stagnation.threshold,
            proportion=self.cfg.genetic.stagnation.proportion,
            cooldown=self.cfg.genetic.stagnation.cooldown,
        )
        islands = [
            Island(
                identity=i,
                ctx=self.ctx,
                n_pop=self.cfg.genetic.parameters.population_size,
                n_offspring=self.cfg.genetic.parameters.offspring_size,
                n_migration=self.cfg.genetic.parameters.migration_size,
                chance_crossover=self.cfg.genetic.parameters.crossover_chance,
                chance_mutation=self.cfg.genetic.parameters.mutation_chance,
                rng=self.cfg.rng,
                opstat=operator_stats,
                fitness_history=fitness_history.copy(),
                stagnation=stagnation,
            )
            for i in range(self.cfg.genetic.parameters.num_islands)
        ]
        return GA(
            ctx=self.ctx,
            obs=_get_observers(self.progress, self.task_id),
            opstat=operator_stats,
            fitness_history=fitness_history,
            gen_arr=np.arange(1, self.cfg.genetic.parameters.generations + 1),
            migrate_gen=_get_migration_generations(self.cfg.genetic.parameters),
            seed_pop=self.seed_pop,
            island_seed_map=seeder.get_island_seed_map(),
            islands=islands,
            start_time=time.perf_counter(),
        )
