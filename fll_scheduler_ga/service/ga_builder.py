"""Builder for the GA instance based on the provided configuration and context."""

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.monitoring import LoggingObserver, RichObserver
from fll_scheduler_ga.genetic.ga import GA, FitnessHistory, OperatorStats, StagnationHandler

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.domain.model import GaObserver, Schedule
    from fll_scheduler_ga.genetic.context import GaContext


logger = logging.getLogger(__name__)


def _get_tracker(operators: tuple) -> dict[str, Counter]:
    inner = {str(o): 0 for o in operators}
    return {"success": Counter(inner), "total": Counter(inner)}


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
    seed_pop: list[Schedule]

    def build(self) -> GA:
        """Build the GA instance."""
        operator_stats = OperatorStats(
            offspring=Counter(),
            crossover=_get_tracker(self.ctx.crossovers),
            mutation=_get_tracker(self.ctx.mutations),
        )
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
        return GA(
            ctx=self.ctx,
            obs=_get_observers(self.progress, self.task_id),
            opstat=operator_stats,
            fitness_history=fitness_history,
            gen_arr=np.arange(1, self.cfg.genetic.parameters.generations + 1),
            start_time=time.perf_counter(),
            rng=self.cfg.rng,
            n_pop=self.cfg.genetic.parameters.population_size,
            n_offspring=self.cfg.genetic.parameters.offspring_size,
            chance_crossover=self.cfg.genetic.parameters.crossover_chance,
            chance_mutation=self.cfg.genetic.parameters.mutation_chance,
            stagnation=stagnation,
            population=self.seed_pop,
        )
