"""Logging module for GA."""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import Counter

    from fll_scheduler_ga.genetic.ga import GA
    from fll_scheduler_ga.genetic.stagnation import OperatorStats
    from fll_scheduler_ga.operators.crossover import Crossover
    from fll_scheduler_ga.operators.mutation import Mutation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GAFinalizer:
    """Finalizer for GA instances."""

    ga: GA

    def __call__(self) -> None:
        """Aggregate islands and run a final selection to produce the final population."""
        self._log_operators(name="crossover", ratios=self.ga.operator_stats.crossover, ops=self.ga.context.crossovers)
        self._log_operators(name="mutation", ratios=self.ga.operator_stats.mutation, ops=self.ga.context.mutations)
        self._log_aggregate_stats(self.ga.operator_stats)
        for island in self.ga.islands:
            logger.debug("Island %d Fitness: %.2f", island.identity, sum(island.fitness_history.get_last_gen_fitness()))
        logger.debug("Total time taken: %.2f seconds", time.perf_counter() - self.ga.start_time)

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
        final_log = f"{'=' * 20}\nFinal statistics"
        crs_suc, crs_tot, crs_rte = operator_stats.get_crossover_stats()
        mut_suc, mut_tot, mut_rte = operator_stats.get_mutation_stats()
        off_suc, off_tot, off_rte = operator_stats.get_offspring_stats()
        unique_inds = len(self.ga.total_population)
        total_inds = len(self.ga)
        unique_rte = f"{unique_inds / total_inds if total_inds > 0 else 0.0:.2%}"
        final_log += (
            f"\n  Total islands          : {len(self.ga.islands)}"
            f"\n  Unique individuals     : {unique_inds}/{total_inds} ({unique_rte})"
            f"\n  Crossover success rate : {crs_suc}/{crs_tot} ({crs_rte})"
            f"\n  Mutation success rate  : {mut_suc}/{mut_tot} ({mut_rte})"
            f"\n  Offspring success rate : {off_suc}/{off_tot} ({off_rte})"
        )
        logger.debug(final_log)
