"""Observers for the FLL Scheduler GA."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import Counter

    import numpy as np
    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.genetic.ga import GA
    from fll_scheduler_ga.genetic.operator import Crossover, Mutation

logger = getLogger(__name__)


class GaObserver(ABC):
    """Abstract base class for observers in the FLL Scheduler GA."""

    @abstractmethod
    def on_start(self, n_generations: int) -> None: ...
    @abstractmethod
    def on_generation_end(self, generation: int, n_generations: int, best_fitness: np.ndarray, npop: int) -> None: ...
    @abstractmethod
    def on_finish(self, npop: int, nfront: int) -> None: ...


class LoggingObserver(GaObserver):
    """Observer that logs generation and best fitness information."""

    def on_start(self, n_generations: int) -> None:
        """Log the start of the genetic algorithm run."""
        logger.debug("Starting genetic algorithm run for %d generations.", n_generations)

    def on_generation_end(self, generation: int, n_generations: int, best_fitness: np.ndarray, npop: int) -> None:
        """Log the end of a generation with population size and best fitness."""
        fit_str = "N/A"
        if best_fitness.any():
            sum_best_fit = best_fitness.sum()
            fit_str = ", ".join([f"{s:.2f}" for s in best_fitness])
            fit_str += f" | Σ={sum_best_fit:.2f} ({sum_best_fit / best_fitness.shape[0]:.1%})"
        logger.debug("Fitness %s | Pop: %d | Generation %d/%d", fit_str, npop, generation, n_generations)

    def on_finish(self, npop: int, nfront: int) -> None:
        """Log the completion of the genetic algorithm run."""
        logger.debug("Genetic algorithm run completed.")
        if not npop:
            logger.warning("No valid schedule was found after all generations.")
            return
        front_portion = nfront / npop * 100 if npop > 0 else 0.0
        logger.debug("Final pareto front size: %d/%d (%.2f%%)", nfront, npop, front_portion)


@dataclass(slots=True)
class RichObserver(GaObserver):
    """Connects GA progress to a Rich Progress Task."""

    progress: Progress
    task_id: TaskID

    def on_start(self, n_generations: int) -> None:
        """Initialize progress task."""
        self.progress.update(task_id=self.task_id, total=n_generations, description="[cyan]Starting...[/cyan]")

    def on_generation_end(self, generation: int, n_generations: int, best_fitness: np.ndarray, npop: int) -> None:
        """Update progress task at generation end."""
        fit_str = "N/A"
        if best_fitness.any():
            sum_best_fit = best_fitness.sum()
            fit_str = ", ".join([f"{s:.3f}" for s in best_fitness])
            fit_str += f" | Σ={sum_best_fit:.3f} ({sum_best_fit / best_fitness.shape[0]:.2%})"
        self.progress.update(
            self.task_id,
            total=n_generations,
            completed=generation,
            description=f"[cyan]Pop: {npop}[/cyan] | [green]Fitness: {fit_str}[/green]",
        )

    def on_finish(self, npop: int, nfront: int) -> None:
        """Finalize progress task."""


def log_ga(ga: GA) -> None:
    """Aggregate islands and run a final selection to produce the final population."""
    _log_operators(name="crossover", ratios=ga.opstat.crossover, ops=ga.context.crossovers)
    _log_operators(name="mutation", ratios=ga.opstat.mutation, ops=ga.context.mutations)
    _log_aggregate_stats(ga)
    for island in ga.islands:
        logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.fitness_history.get_last_gen_fitness()))
    logger.debug("Total time taken: %.2f seconds", time.perf_counter() - ga.start_time)


def _log_operators(name: str, ratios: dict[str, Counter], ops: tuple[Crossover | Mutation, ...]) -> None:
    """Log statistics for crossover and mutation operators."""
    if not (op_strings := [f"{o!s}" for o in ops]):
        return
    log = f"{name.capitalize()} statistics:"
    max_len = max(len(s) for s in op_strings) + 1
    for o in op_strings:
        success = ratios.get("success", {}).get(o, 0)
        total = ratios.get("total", {}).get(o, 0)
        rate = success / total if total > 0 else 0.0
        log += f"\n  {o:<{max_len}}: {success}/{total} ({rate:.2%})"
    logger.debug(log)


def _log_aggregate_stats(ga: GA) -> None:
    """Log aggregate statistics across all islands."""
    final_log = f"{'=' * 20}\nFinal statistics"
    crs_suc, crs_tot, crs_rte = ga.opstat.get_crossover_stats()
    mut_suc, mut_tot, mut_rte = ga.opstat.get_mutation_stats()
    off_suc, off_tot, off_rte = ga.opstat.get_offspring_stats()
    unique_inds = len(ga.total_pop)
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
