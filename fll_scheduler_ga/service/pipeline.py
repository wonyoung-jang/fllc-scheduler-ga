"""Main pipeline."""

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import CsvScheduleExporter, ScheduleSummaryGenerator, generate_summary
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.adapter.logger import log_ga
from fll_scheduler_ga.adapter.observer import GaObserver, LoggingObserver, RichObserver
from fll_scheduler_ga.adapter.plot import MatplotlibVisualizer
from fll_scheduler_ga.adapter.schema import GaParameterModel, build_app_config, log_appconfig_creation_info
from fll_scheduler_ga.adapter.seeder import GASeedData, GASeeder, load_ga, save_ga
from fll_scheduler_ga.genetic.context import build_ga_context
from fll_scheduler_ga.genetic.ga import GA, FitnessHistory, OperatorStats

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.domain.schedule import Schedule
    from fll_scheduler_ga.genetic.context import GaContext

logger = logging.getLogger(__name__)


def _get_tracker(operators: tuple) -> dict[str, Counter]:
    inner = {str(o): 0 for o in operators}
    return {"success": Counter(inner), "total": Counter(inner)}


def _run_runtime_startup(config: AppConfig, context: GaContext) -> None:
    """Run the import schedule handler."""
    seed_file = Path(config.runtime.seed_file).resolve()
    if config.runtime.flush and seed_file.exists():
        seed_file.unlink(missing_ok=True)
        seed_file.touch(exist_ok=True)
        logger.debug("Flushed seed file at: %s", seed_file)
    importsched = _import(config, context)
    if importsched is not None and config.runtime.add_import_to_population:
        population = load_ga(path=seed_file, config=config.tournament)
        if importsched not in population:
            population.append(importsched)
        save_ga(seed_file, GASeedData(config.tournament, population))


def _import(config: AppConfig, context: GaContext) -> Schedule | None:
    """Handle the import file for the genetic algorithm."""
    if not config.runtime.import_file:
        logger.debug("No import file specified, skipping import step.")
        return None
    import_path = Path(config.runtime.import_file).resolve()
    csv_importer = CsvImporter(import_path, config.tournament, context.event_factory, context.event_properties)
    if not csv_importer.validate_inputs():
        return None
    csv_importer.run()
    imported_schedule = csv_importer.schedule
    if not context.check(imported_schedule):
        context.repair(imported_schedule)
    if fits := context.evaluate(np.array([imported_schedule.schedule], dtype=int)):
        sched_fits, team_fits = fits
        imported_schedule.fitness = sched_fits
        imported_schedule.team_fitnesses = team_fits
        parent_dir = import_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        report_path = parent_dir / "report.txt"
        team_ids = config.io.exports.team_identities
        ScheduleSummaryGenerator(team_ids).export(imported_schedule, report_path)
        CsvScheduleExporter(
            time_fmt=context.tournament_config.time_fmt,
            team_identities=team_ids,
            event_properties=context.event_properties,
        ).export(imported_schedule, parent_dir / "schedule.csv")
    return imported_schedule


def _get_migration_generations(gp: GaParameterModel) -> np.ndarray:
    """Calculate the generations at which migration should occur."""
    migrate_generations = np.zeros(gp.generations + 1, dtype=int)
    if gp.num_islands > 1 and gp.migration_size > 0:
        migrate_generations[:: gp.migration_interval] = 1
    return migrate_generations


def _get_observers(progress: Progress | None, task_id: TaskID | None) -> tuple[GaObserver, ...]:
    """Get the observers for the GA."""
    observers: list[GaObserver] = [LoggingObserver()]
    if progress and task_id is not None:
        observers.append(RichObserver(progress, task_id))
    return tuple(observers)


def _finalize_ga_results(cfg: AppConfig, ctx: GaContext, seed_file: Path, ga: GA) -> None:
    """Finalize the GA results by exporting schedules and generating summaries."""
    log_ga(ga)
    data = GASeedData(ctx.tournament_config, ga.pareto_front if cfg.io.exports.front_only else ga.total_population)
    save_ga(seed_file, data)
    output_dir = Path(cfg.io.exports.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = MatplotlibVisualizer(
        ga.total_population, ga.fitness_history.history, output_dir, ctx.nsga3.refs.points, cfg.io.exports
    )
    generate_summary(ga, output_dir, cfg.io.exports, plot)


def run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    cfg = build_app_config(config_path)
    log_appconfig_creation_info(cfg)
    ctx = build_ga_context(cfg)
    _run_runtime_startup(cfg, ctx)
    seed_file = Path(cfg.runtime.seed_file).resolve()
    seed_pop = load_ga(seed_file, ctx.tournament_config)
    ga = GA(
        context=ctx,
        genetic_model=cfg.genetic,
        rng=cfg.rng,
        observers=_get_observers(progress, task_id),
        operator_stats=OperatorStats(
            offspring=Counter(), crossover=_get_tracker(ctx.crossovers), mutation=_get_tracker(ctx.mutations)
        ),
        fitness_history=FitnessHistory(
            generation=0,
            current=np.zeros((1, ctx.evaluator.n_objectives), dtype=float),
            history=np.full(
                (cfg.genetic.parameters.generations, ctx.evaluator.n_objectives), fill_value=-1, dtype=float
            ),
        ),
        generations_array=np.arange(1, cfg.genetic.parameters.generations + 1),
        migrate_generations=_get_migration_generations(cfg.genetic.parameters),
        seed_pop=seed_pop,
        island_seed_map=GASeeder(
            rng=cfg.rng,
            seed_pop_size=len(seed_pop),
            seed_island_strategy=ctx.seed_island_strategy,
            seed_pop_sort=ctx.seed_pop_sort,
            n_islands=cfg.genetic.parameters.num_islands,
            n_pop=cfg.genetic.parameters.population_size,
        ).get_island_seed_map(),
    )
    ga.run()
    _finalize_ga_results(cfg, ctx, seed_file, ga)
    return ga
