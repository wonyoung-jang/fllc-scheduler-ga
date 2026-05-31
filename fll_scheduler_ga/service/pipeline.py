"""Main pipeline."""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.exporter import CsvScheduleExporter, ScheduleSummaryGenerator, generate_summary
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.adapter.monitoring import GaObserver, LoggingObserver, RichObserver, log_ga
from fll_scheduler_ga.adapter.plot import MatplotlibVisualizer
from fll_scheduler_ga.adapter.schema import GaParameterModel, build_app_config, log_appconfig_creation_info
from fll_scheduler_ga.adapter.seeder import load_fitness_benchmark, load_ga, save_fitness_benchmark, save_ga
from fll_scheduler_ga.constants import BENCHMARKS_CACHE, FitnessObjective, SeedIslandStrategy, SeedPopSort
from fll_scheduler_ga.domain.model import (
    BenchmarkSeedData,
    EventFactory,
    EventProperties,
    GASeedData,
    build_event_factory,
    build_event_props,
)
from fll_scheduler_ga.domain.schedule import Schedule, ScheduleContext
from fll_scheduler_ga.genetic.context import GaContext, ScheduleBuilderRandom
from fll_scheduler_ga.genetic.fitness import (
    FitnessBenchmarkBreaktime,
    FitnessBenchmarkOpponent,
    FitnessEvaluator,
    generate_stable_config_hash,
)
from fll_scheduler_ga.genetic.ga import GA, FitnessHistory, OperatorStats
from fll_scheduler_ga.genetic.operator import (
    NSGA3,
    NonDominatedSorting,
    RandomSelect,
    ReferenceDirections,
    Repairer,
    build_crossovers,
    build_mutations,
    calc_norm_sq_of_refs,
    calc_ref_points,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.progress import Progress, TaskID

    from fll_scheduler_ga.adapter.schema import AppConfig
    from fll_scheduler_ga.domain.model import TimeSlot

logger = logging.getLogger(__name__)


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
    if not config.runtime.import_file:
        logger.debug("No import file specified, skipping import step.")
        return
    importsched = _import(config, context)
    if importsched is not None and config.runtime.add_import_to_population:
        population = load_ga(path=seed_file, config=config.tournament)
        if importsched not in population:
            population.append(importsched)
        save_ga(seed_file, GASeedData(config.tournament, population))


def _import(config: AppConfig, context: GaContext) -> Schedule | None:
    """Handle the import file for the genetic algorithm."""
    import_path = Path(config.runtime.import_file).resolve()
    csv_importer = CsvImporter(import_path, config.tournament, context.event_factory, context.event_properties)
    if not csv_importer.validate_inputs():
        return None
    csv_importer.run()
    importsched = csv_importer.schedule
    if not context.check(importsched):
        context.repair(importsched)
    if fits := context.evaluate(np.array([importsched.schedule], dtype=int)):
        sched_fits, team_fits = fits
        importsched.fitness = sched_fits
        importsched.team_fitnesses = team_fits
        parent_dir = import_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        report_path = parent_dir / "report.txt"
        team_ids = config.io.exports.team_identities
        ScheduleSummaryGenerator(team_ids).export(importsched, report_path)
        CsvScheduleExporter(
            time_fmt=context.tournament_config.time_fmt,
            team_identities=team_ids,
            event_properties=context.event_properties,
        ).export(importsched, parent_dir / "schedule.csv")
    return importsched


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


def distributed_seeding(seed_indices: Iterator[int], n_islands: int) -> dict[int, list[int]]:
    """Get the seed indices for each island."""
    island_to_seed: dict[int, list[int]] = defaultdict(list)
    for idx in seed_indices:
        island_to_seed[idx % n_islands].append(idx)
    return island_to_seed


def concentrated_seeding(seed_indices: Iterator[int], n_islands: int, n_pop: int) -> dict[int, list[int]]:
    """Get the seed indices for each island."""
    island_to_seed: dict[int, list[int]] = defaultdict(list)
    for i in range(n_islands):
        while len(island_to_seed[i]) < n_pop:
            if (idx := next(seed_indices, None)) is None:
                break
            island_to_seed[i].append(idx)
    return island_to_seed


@dataclass(slots=True)
class GASeeder:
    """Seeding strategies for GA instances."""

    rng: np.random.Generator
    seed_pop_size: int
    seed_island_strategy: str
    seed_pop_sort: str
    n_islands: int
    n_pop: int

    def is_valid(self) -> bool:
        """Check if seeding is valid based on the provided seed population."""
        if not self.seed_pop_size:
            logger.debug("No seed population provided. Starting with a fresh population.")
            return False
        logger.debug("Seeding population with %d individuals from seed file.", self.seed_pop_size)
        logger.debug("Seed pop sort: %s | Seed island strategy: %s", self.seed_pop_sort, self.seed_island_strategy)
        return True

    def get_island_seed_map(self) -> dict[int, list[int]]:
        """Get the mapping of islands to seed indices based on the seeding strategy."""
        if not self.is_valid():
            return {}
        match self.seed_island_strategy:
            case SeedIslandStrategy.CONCENTRATED:
                return concentrated_seeding(self.iter_seeds(), self.n_islands, self.n_pop)
            case SeedIslandStrategy.DISTRIBUTED:
                return distributed_seeding(self.iter_seeds(), self.n_islands)
            case _:
                return distributed_seeding(self.iter_seeds(), self.n_islands)

    def iter_seeds(self) -> Iterator[int]:
        """Yield indices for seeding strategies."""
        match self.seed_pop_sort:
            case SeedPopSort.BEST:
                yield from np.arange(self.seed_pop_size)
            case SeedPopSort.RANDOM:
                yield from self.rng.permutation(self.seed_pop_size)
            case _:
                yield from self.rng.permutation(self.seed_pop_size)


def _run_preflight_checks(props: EventProperties, factory: EventFactory) -> None:
    """Run all pre-flight checks."""
    try:
        _check_location_timeslot_overlaps(props, factory)
        logger.debug("All preflight checks passed successfully.")
    except ValueError:
        logger.exception("Preflight checks failed. Please review the configuration.")
        raise


def _check_location_timeslot_overlaps(props: EventProperties, factory: EventFactory) -> None:
    """Check if different round types are scheduled in the same locations at the same time."""
    booked: dict[int, list[tuple[TimeSlot, str]]] = defaultdict(list)
    for e in factory.events_idx:
        loc_str = props.loc_str[e]
        loc_idx = props.loc_idx[e]
        ts = props.timeslot[e]
        rt = props.roundtype[e]
        for existing_ts, existing_rt in booked.get(loc_idx, []):
            if ts.overlaps(existing_ts):
                msg = (
                    f"Configuration conflict: TournamentRound '{rt}' and '{existing_rt}' "
                    f"are scheduled in the same location ({loc_str} {loc_idx}) "
                    f"at overlapping times ({ts} and "
                    f"{existing_ts})."
                )
                raise ValueError(msg)
        booked[loc_idx].append((ts, rt))
    logger.debug("Check passed: No location/time overlaps found.")


def _hard_constraint_checker(constraints: tuple[Callable[[Schedule], bool], ...]) -> Callable[[Schedule], bool]:
    """Check hard constraints of a schedule."""
    return lambda s: not any(constraint(s) for constraint in constraints)


def build_ga_context(cfg: AppConfig) -> GaContext:
    """Build and return a GA context."""
    evt_factory = build_event_factory(cfg.tournament.rounds)
    evt_props = build_event_props(evt_factory.mapping)
    _run_preflight_checks(evt_props, evt_factory)
    Schedule.ctx = ScheduleContext(
        conflict_map=evt_factory.conflict_map,
        event_props=evt_props,
        teams_list=np.arange(cfg.tournament.num_teams, dtype=int),
        teams_roundreqs_arr=np.tile(A=tuple(cfg.tournament.roundreqs.values()), reps=(cfg.tournament.num_teams, 1)),
        empty_schedule=np.full(cfg.tournament.n_total_events, -1, dtype=int),
    )
    constraints = (
        lambda s: not s,
        lambda s: s.get_size() != cfg.tournament.total_slots_required,
        lambda s: s.any_rounds_needed(),
    )
    checker = _hard_constraint_checker(constraints)

    config_hash = generate_stable_config_hash(config=cfg.tournament, model=cfg.fitness)
    BENCHMARKS_CACHE.mkdir(parents=True, exist_ok=True)
    benchmark_path = BENCHMARKS_CACHE / f"benchmark_cache_{config_hash}.pkl"
    benchmark_data = load_fitness_benchmark(benchmark_path)
    if not cfg.runtime.flush_benchmarks and benchmark_data:
        opponents = benchmark_data.opponents
        best_timeslot_score = benchmark_data.best_timeslot_score
    else:
        logger.info("Calculating new benchmarks...")
        opponents = FitnessBenchmarkOpponent(cfg.tournament, evt_factory).benchmark()
        best_timeslot_score = FitnessBenchmarkBreaktime(cfg.tournament, evt_factory, cfg.fitness).benchmark()
        save_fitness_benchmark(benchmark_path, BenchmarkSeedData(opponents, best_timeslot_score))

    points = calc_ref_points(len(FitnessObjective), cfg.genetic.parameters.population_size)
    ref_directions = ReferenceDirections(points.shape[0], points, calc_norm_sq_of_refs(points))

    return GaContext(
        app_config=cfg,
        event_factory=evt_factory,
        event_properties=evt_props,
        builder=ScheduleBuilderRandom(evt_props, cfg.rng, cfg.tournament.round_idx_to_tpr, evt_factory.roundtypes),
        repairer=Repairer(cfg.tournament, evt_factory, evt_props, cfg.rng, checker),
        evaluator=FitnessEvaluator(cfg.tournament, evt_props, cfg.fitness, opponents, best_timeslot_score),
        checker=checker,
        nsga3=NSGA3(cfg.rng, ref_directions, NonDominatedSorting()),
        selection=RandomSelect(cfg.rng),
        crossovers=build_crossovers(cfg.rng, cfg.genetic.operator, evt_factory, evt_props),
        mutations=build_mutations(cfg.rng, cfg.genetic.operator, evt_factory, evt_props),
    )
