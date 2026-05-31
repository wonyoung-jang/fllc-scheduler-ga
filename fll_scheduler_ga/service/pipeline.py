"""Main pipeline."""

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import generate_summary
from fll_scheduler_ga.adapter.logger import GAFinalizer
from fll_scheduler_ga.adapter.observer import LoggingObserver, RichObserver
from fll_scheduler_ga.adapter.plot import MatplotlibVisualizer
from fll_scheduler_ga.adapter.seeder import GASeedData, GASeeder, RuntimeStartup, load_ga, save_ga
from fll_scheduler_ga.constants import FitnessObjective
from fll_scheduler_ga.domain.schema import build_app_config, log_appconfig_creation_info
from fll_scheduler_ga.genetic.context import build_ga_context
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.genetic.stagnation import FitnessHistory, OperatorStats

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


def _get_tracker(operators: tuple) -> dict[str, Counter]:
    inner = {str(o): 0 for o in operators}
    return {"success": Counter(inner), "total": Counter(inner)}


def run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    cfg = build_app_config(config_path)
    log_appconfig_creation_info(cfg)
    ctx = build_ga_context(cfg)
    RuntimeStartup(config=cfg, context=ctx).run()
    n_gen = cfg.genetic.parameters.generations
    n_obj = ctx.evaluator.n_objectives
    migrate_generations = np.zeros(n_gen + 1, dtype=int)
    n_islands = cfg.genetic.parameters.num_islands
    migration_size = cfg.genetic.parameters.migration_size
    migration_interval = cfg.genetic.parameters.migration_interval
    if n_islands > 1 and migration_size > 0:
        migrate_generations[::migration_interval] = 1
    export_model = cfg.io.exports
    seed_file = Path(cfg.runtime.seed_file).resolve()
    pre_seed_data = load_ga(seed_file, ctx.tournament_config)
    seed_pop = pre_seed_data.population if pre_seed_data else []
    observers: list = [LoggingObserver()]
    if progress and task_id is not None:
        observers.append(RichObserver(progress, task_id))
    ga = GA(
        context=ctx,
        genetic_model=cfg.genetic,
        rng=cfg.rng,
        observers=tuple(observers),
        operator_stats=OperatorStats(
            offspring=Counter(), crossover=_get_tracker(ctx.crossovers), mutation=_get_tracker(ctx.mutations)
        ),
        fitness_history=FitnessHistory(
            generation=0,
            current=np.zeros((1, n_obj), dtype=float),
            history=np.full((n_gen, n_obj), fill_value=-1, dtype=float),
        ),
        generations_array=np.arange(1, n_gen + 1),
        migrate_generations=migrate_generations,
        seed_pop=seed_pop,
        island_seed_map=GASeeder(
            imports=ctx.import_model,
            seed_pop=seed_pop,
            rng=cfg.rng,
            seed_island_strategy=ctx.seed_island_strategy,
            n_islands=n_islands,
            pop_size=cfg.genetic.parameters.population_size,
        ).get_island_seed_map(),
    )
    ga.run()
    GAFinalizer(ga)()
    post_seed_data = GASeedData(
        ctx.tournament_config, ga.pareto_front if export_model.front_only else ga.total_population
    )
    save_ga(seed_file, post_seed_data)
    output_dir = Path(export_model.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = MatplotlibVisualizer(
        total_population=ga.total_population,
        fitness_history=ga.fitness_history.history,
        save_dir=output_dir,
        objectives=tuple(FitnessObjective),
        ref_points=ga.context.nsga3.refs.points,
        export_model=export_model,
    )
    generate_summary(ga=ga, output_dir=output_dir, export_model=export_model, plot=plot)
    return ga
