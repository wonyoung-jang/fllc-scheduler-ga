"""Typer based CLI scheduler implementation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config.app_config import AppConfig
from ..config.constants import FitnessObjective
from ..genetic.ga import GA
from ..genetic.ga_context import StandardGaContextFactory
from ..genetic.ga_generation import GaGeneration
from ..genetic.stagnation import FitnessHistory, OperatorStats
from ..io import ga_exporter
from ..io.observers import LoggingObserver, RichObserver
from ..io.plot import MatplotlibVisualizer

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


def run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    app_config = AppConfig.build(config_path)
    app_config.log_creation_info()
    context_factory = StandardGaContextFactory()
    context = context_factory.build(app_config)

    trackers = ("success", "total")
    crossover_counters = {str(c): 0 for c in context.crossovers}
    mutation_counters = {str(m): 0 for m in context.mutations}
    operator_stats = OperatorStats(
        offspring=Counter(),
        crossover={tr: Counter(crossover_counters) for tr in trackers},
        mutation={tr: Counter(mutation_counters) for tr in trackers},
    )

    n_gen = app_config.genetic.parameters.generations
    n_obj = context.evaluator.n_objectives
    generation = GaGeneration(curr=0)
    fitness_history = FitnessHistory(
        generation=generation,
        current=np.zeros((1, n_obj), dtype=float),
        history=np.full((n_gen, n_obj), fill_value=-1, dtype=float),
    )

    generations_array = np.arange(1, n_gen + 1)
    migrate_generations = np.zeros(n_gen + 1, dtype=int)
    n_islands = app_config.genetic.parameters.num_islands
    migration_size = app_config.genetic.parameters.migration_size
    migration_interval = app_config.genetic.parameters.migration_interval
    if n_islands > 1 and migration_size > 0:
        migrate_generations[::migration_interval] = 1

    ga = GA(
        context=context,
        genetic_model=app_config.genetic,
        rng=app_config.rng,
        observers=(LoggingObserver(),),
        seed_file=Path(app_config.runtime.seed_file),
        save_front_only=app_config.exports.front_only,
        generation=generation,
        operator_stats=operator_stats,
        fitness_history=fitness_history,
        generations_array=generations_array,
        migrate_generations=migrate_generations,
    )

    if progress and task_id is not None:
        existing = list(ga.observers)
        existing.append(RichObserver(progress, task_id))
        ga.observers = tuple(existing)

    ga.run()
    exports = app_config.exports
    output_dir = Path(exports.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot = MatplotlibVisualizer(
        ga=ga,
        save_dir=output_dir,
        objectives=tuple(FitnessObjective),
        ref_points=ga.context.nsga3.refs.points,
        export_model=exports,
    )
    ga_exporter.generate_summary(
        ga=ga,
        output_dir=output_dir,
        export_model=exports,
        plot=plot,
    )

    return ga
