"""Main pipeline."""

import datetime as dt
import hashlib
import itertools
import logging
import math
import pprint as pp
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.adapter.exporter import (
    CsvScheduleExporter,
    MatplotlibVisualizer,
    ScheduleSummaryGenerator,
    SummaryManager,
)
from fll_scheduler_ga.adapter.importer import CsvImporter
from fll_scheduler_ga.adapter.monitoring import GaObserver, LoggingObserver, RichObserver, log_ga
from fll_scheduler_ga.adapter.schema import (
    AppConfig,
    AppConfigModel,
    FitnessModel,
    GaParameterModel,
    LocationModel,
    RoundModel,
    build_app_config_model,
)
from fll_scheduler_ga.adapter.seeder import load_fitness_benchmark, load_ga, save_fitness_benchmark, save_ga
from fll_scheduler_ga.constants import (
    BENCHMARKS_CACHE,
    RANDOM_SEED_RANGE,
    FitnessObjective,
    SeedIslandStrategy,
    SeedPopSort,
)
from fll_scheduler_ga.domain.model import (
    DEFAULT_DT,
    BenchmarkSeedData,
    Event,
    EventProperties,
    EventRepository,
    GASeedData,
    Location,
    Schedule,
    ScheduleContext,
    TimeSlot,
    TournamentConfig,
    TournamentRound,
)
from fll_scheduler_ga.genetic.context import GaContext, ScheduleBuilderRandom
from fll_scheduler_ga.genetic.fitness import FitnessBenchmarkBreaktime, FitnessBenchmarkOpponent, FitnessEvaluator
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
    from collections.abc import Callable, Iterable, Iterator

    from numpy.random import Generator
    from rich.progress import Progress, TaskID


logger = logging.getLogger(__name__)

TIME_FORMAT_MAP = {12: "%I:%M %p", 24: "%H:%M"}


def _parse_time_str(dt_str: str, fmt: str) -> dt.datetime:
    """Parse a time string into a datetime object."""
    if not dt_str:
        return DEFAULT_DT
    return dt.datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=dt.UTC)


def _infer_time_format(dt_str: str) -> str | None:
    """Infer the time format from a sample time string."""
    for fmt in TIME_FORMAT_MAP.values():
        try:
            _parse_time_str(dt_str, fmt)
        except ValueError:
            continue
        return fmt
    return None


def _calc_num_timeslots(n_locs: int, n_teams: int, rounds_per_team: int) -> int:
    """Calculate the number of timeslots needed for a round."""
    if n_locs <= 0:
        msg = "Cannot calculate number of timeslots without times or locations."
        raise ValueError(msg)
    return math.ceil((n_teams * rounds_per_team) / n_locs)


def _validate_duration(
    start_stop: tuple[dt.datetime, dt.datetime], times_dt: tuple[dt.datetime, ...], dur: int, n_timeslots: int
) -> dt.timedelta:
    """Validate the times configuration for a round.

    Valid conditions:
    1. start + duration
    2. times + duration
    3. start + stop (need to calculate num_timeslots).
    """
    start_dt, stop_dt = start_stop
    if (start_dt != DEFAULT_DT or times_dt) and dur:
        return dt.timedelta(minutes=dur)
    diff = stop_dt - start_dt
    total_available = diff.total_seconds()
    minimum_duration = total_available // n_timeslots
    return dt.timedelta(minutes=max(1, minimum_duration // 60))


def _init_timeslots(
    starts: tuple[dt.datetime, ...],
    dur_cycle: dt.timedelta,
    dur_active: dt.timedelta,
    n_timeslots: int,
    start_dt: dt.datetime,
) -> Iterator[tuple[dt.datetime, ...]]:
    """Initialize the timeslots for the round."""
    if starts and dur_active and dur_cycle:
        stops_cycle = (*starts[1:], starts[-1] + dur_cycle)
        stops_active = (start + dur_active for start in starts)
        yield from zip(starts, stops_active, stops_cycle, strict=True)
    else:
        curr = start_dt
        for _ in range(n_timeslots):
            stop_cycle = curr + dur_cycle
            stop_active = curr + dur_active
            yield (curr, stop_active, stop_cycle)
            curr = stop_cycle


def parse_teams(teams: tuple[int | str, ...] | int) -> tuple[str, ...]:
    """Return a tuple of team identifiers."""
    if isinstance(teams, int):
        return tuple(str(i) for i in range(1, teams + 1))
    return tuple(str(t) for t in teams)


def get_rng(seed: int | str | None) -> Generator:
    """Return the RNG seed as an integer."""
    if isinstance(seed, int):
        s = seed
    elif seed is None:
        s = int(np.random.default_rng().integers(*RANDOM_SEED_RANGE))
    else:
        s = abs(hash(seed)) % (RANDOM_SEED_RANGE[1] + 1)
    return np.random.default_rng(s)


def parse_locations(location_model: tuple[LocationModel, ...]) -> tuple[Location, ...]:
    """Parse location models into Location instances."""

    def _generate_locations(idx_iter: Iterator[int]) -> Iterator[Location]:
        for lm in location_model:
            for name in range(1, lm.count + 1):
                for side in range(1, lm.sides + 1):
                    yield Location(next(idx_iter), lm.name, name, -1 if lm.sides == 1 else side, lm.sides)

    idx_iter = itertools.count()
    return tuple(_generate_locations(idx_iter))


def get_tournament_config(n_teams: int, time_fmt: str, rounds: tuple[TournamentRound, ...]) -> TournamentConfig:
    """Load and return the tournament configuration from the validated model."""
    roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
    return TournamentConfig(
        num_teams=n_teams,
        time_fmt=time_fmt,
        rounds=rounds,
        roundreqs=roundreqs,
        round_idx_to_tpr={r.roundtype_idx: r.teams_per_round for r in rounds},
        total_slots_required=sum(r.slots_required for r in rounds),
        unique_opponents_possible=1 <= max(roundreqs.values()) <= n_teams - 1,
        max_events_per_team=sum(roundreqs.values()),
        all_locations=_get_all_sorted_attr(rounds, get_by="locations", sort_by="idx"),
        all_timeslots=_get_all_sorted_attr(rounds, get_by="timeslots", sort_by="idx"),
        is_interleaved=_are_rounds_overlapping(rounds),
    )


def _generate_all_time_strs(round_models: Iterable[RoundModel]) -> Iterator[str]:
    for rm in round_models:
        yield from (rm.start_time, rm.stop_time, *rm.times)


def parse_time_fmt(round_models: Iterable[RoundModel]) -> str:
    """Get the time format from the rounds configuration."""
    inference = (_infer_time_format(t) for t in _generate_all_time_strs(round_models) if t)
    valid = [fmt for fmt in inference if fmt is not None]
    if not valid:
        msg = "No time strings found in configuration to infer time format."
        raise ValueError(msg)
    format_counts = Counter(valid)
    if len(format_counts) != 1:
        msg = "Conflicting time formats found in configuration times."
        raise ValueError(msg)
    return str(format_counts.most_common(1)[0][0])


def parse_rounds(
    models: tuple[RoundModel, ...], n_teams: int, time_fmt: str, all_locations: tuple[Location, ...]
) -> tuple[TournamentRound, ...]:
    """Parse and return TournamentRound objects from the configuration."""

    def _generate_rounds(timeslot_idx_iter: Iterator[int]) -> Iterator[TournamentRound]:
        for roundtype_idx, rm in enumerate(models):
            locations = tuple(loc for loc in all_locations if loc.locationtype == rm.location)
            nloc = len(locations)
            start_dt = _parse_time_str(rm.start_time, time_fmt)
            stop_dt = _parse_time_str(rm.stop_time, time_fmt)
            _input_times_dt = tuple(_parse_time_str(t, time_fmt) for t in rm.times) if rm.times else ()
            n_timeslot = len(_input_times_dt) or _calc_num_timeslots(nloc, n_teams, rm.rounds_per_team)
            if n_timeslot <= 0:
                msg = "n_timeslots must be greater than zero to validate duration."
                raise ValueError(msg)
            start_stop = (start_dt, stop_dt)
            dur_tdelta_cycle = _validate_duration(start_stop, _input_times_dt, rm.duration_cycle, n_timeslot)
            dur_tdelta_active = _validate_duration(start_stop, _input_times_dt, rm.duration_active, n_timeslot)
            timeslots_iter = _init_timeslots(_input_times_dt, dur_tdelta_cycle, dur_tdelta_active, n_timeslot, start_dt)
            timeslots = tuple(
                TimeSlot(next(timeslot_idx_iter), start, stop_active, stop_cycle)
                for start, stop_active, stop_cycle in timeslots_iter
            )
            round_start_time = timeslots[0].start
            round_stop_time = timeslots[-1].stop_cycle
            times_dt = tuple(ts.start for ts in timeslots)
            slots_total = n_timeslot * nloc
            slots_required = n_teams * rm.rounds_per_team
            slots_empty = slots_total - slots_required
            if slots_empty < 0:
                msg = (
                    "Insufficient capacity for TournamentRound (required > available).\n"
                    "Suggestion: increase number of locations or timeslots."
                )
                raise ValueError(msg)
            unfilled_allowed = slots_empty > 0
            yield TournamentRound(
                rm.roundtype,
                roundtype_idx,
                rm.rounds_per_team,
                rm.teams_per_round,
                times_dt,
                round_start_time,
                round_stop_time,
                dur_tdelta_cycle,
                rm.location,
                locations,
                n_timeslot,
                timeslots,
                slots_total,
                slots_required,
                slots_empty,
                unfilled_allowed,
            )

    timeslot_idx_iter = itertools.count()
    return tuple(sorted(_generate_rounds(timeslot_idx_iter), key=lambda r: r.start_time))


def _get_all_sorted_attr(objects: Iterable[Any], get_by: str, sort_by: str) -> tuple[Any, ...]:
    """Get all attributes of the TournamentRound objects."""
    return tuple(
        sorted(itertools.chain.from_iterable(getattr(r, get_by) for r in objects), key=lambda x: getattr(x, sort_by))
    )


def _are_rounds_overlapping(rounds: Iterable[TournamentRound]) -> bool:
    """Check if any rounds are interleaved in time."""
    start_stops = ((r.start_time, r.stop_time) for r in rounds)
    timeslots = (TimeSlot(idx=0, start=start, stop_active=stop, stop_cycle=stop) for start, stop in start_stops)
    return any(a.overlaps(b) for a, b in itertools.combinations(timeslots, 2))


def build_app_config(m: AppConfigModel) -> AppConfig:
    """Build and return the application configuration."""
    teams = parse_teams(m.tournament.teams)
    team_identities = dict(enumerate(teams, start=1))
    if not (locations := parse_locations(m.tournament.locations)):
        msg = "No locations defined in the configuration file."
        raise ValueError(msg)
    time_fmt = parse_time_fmt(m.tournament.rounds)
    TimeSlot.time_fmt = time_fmt
    if not (rounds := parse_rounds(m.tournament.rounds, len(teams), time_fmt, locations)):
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)
    tournament_config = get_tournament_config(len(teams), time_fmt, rounds)
    rng = get_rng(m.genetic.rng_seed)
    cfg = AppConfig(m.genetic, m.runtime, m.io, m.fitness, tournament_config, team_identities, rng)
    logger.debug("Initialized AppConfig: %s", pp.pformat(cfg))
    return cfg


def _get_tracker(operators: tuple) -> dict[str, Counter]:
    inner = {str(o): 0 for o in operators}
    return {"success": Counter(inner), "total": Counter(inner)}


def _import_schedule(cfg: AppConfig, ctx: GaContext) -> None:
    """Run the import schedule handler."""
    path = Path(cfg.runtime.seed_file).resolve()
    if cfg.runtime.flush and path.exists():
        path.unlink(missing_ok=True)
        path.touch(exist_ok=True)
        logger.debug("Flushed seed file at: %s", path)
    if not cfg.runtime.import_file:
        logger.debug("No import file specified, skipping import step.")
        return
    importsched = _import(cfg, ctx)
    if importsched is not None and cfg.runtime.add_import_to_population:
        population = load_ga(path=path, config=cfg.tournament)
        if importsched not in population:
            population.append(importsched)
        save_ga(path, GASeedData(cfg.tournament, population))


def _import(cfg: AppConfig, ctx: GaContext) -> Schedule | None:
    """Handle the import file for the genetic algorithm."""
    path = Path(cfg.runtime.import_file).resolve()
    csv_importer = CsvImporter(path, cfg.tournament, ctx.event_repo, ctx.event_properties)
    if not csv_importer.validate_inputs():
        return None
    csv_importer.run()
    importsched = csv_importer.sched
    if not ctx.check(importsched):
        ctx.repair(importsched)
    if fits := ctx.evaluate(np.array([importsched.schedule], dtype=int)):
        sched_fits, team_fits = fits
        importsched.fitness = sched_fits
        importsched.team_fitnesses = team_fits
        parent_dir = path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        report_path = parent_dir / "report.txt"
        ScheduleSummaryGenerator(cfg.team_identities).export(importsched, report_path)
        CsvScheduleExporter(
            time_fmt=cfg.tournament.time_fmt, team_identities=cfg.team_identities, event_properties=ctx.event_properties
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
    data = GASeedData(cfg.tournament, ga.pareto_front if cfg.io.exports.front_only else ga.total_population)
    save_ga(seed_file, data)
    outdir = Path(cfg.io.exports.output_dir).resolve()
    if outdir.exists():
        logger.debug("Output directory %s already exists. Clearing contents.", outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot = (
        MatplotlibVisualizer(
            total_population=ga.total_population,
            fitness_history=ga.fitness_history.history,
            save_dir=outdir,
            ref_points=ctx.nsga3.refs.points,
            cmap_name=cfg.io.exports.cmap_name,
            is_plot_fitness=cfg.io.exports.plot_fitness,
            is_plot_parallel=cfg.io.exports.plot_parallel,
            is_plot_scatter=cfg.io.exports.plot_scatter,
        )
        if not cfg.io.exports.no_plotting and ga.total_population
        else None
    )
    summarizer = SummaryManager(
        outdir,
        plot=plot,
        export_pareto_summary=cfg.io.exports.pareto_summary,
        export_schedules_csv=cfg.io.exports.schedules_csv,
        export_schedules_html=cfg.io.exports.schedules_html,
        export_summary_reports=cfg.io.exports.summary_reports,
        export_schedules_team_csv=cfg.io.exports.schedules_team_csv,
        export_front_only=cfg.io.exports.front_only,
        pareto_front=ga.pareto_front,
        total_population=ga.total_population,
        roundreqs=cfg.tournament.roundreqs,
        time_fmt=cfg.tournament.time_fmt,
        team_ids=cfg.team_identities,
        evt_prop=ctx.event_properties,
    )
    summarizer.generate()


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


def _run_preflight_checks(prop: EventProperties, repo: EventRepository) -> None:
    """Run all pre-flight checks.

    Check if different round types are scheduled in the same locations at the same time.
    """
    booked: dict[int, list[tuple[TimeSlot, str]]] = defaultdict(list)
    for e in repo.events_idx:
        loc_str = prop.loc_str[e]
        loc_idx = prop.loc_idx[e]
        ts = prop.timeslot[e]
        rt = prop.roundtype[e]
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
    logger.debug("All preflight checks passed successfully.")


def _hard_constraint_checker(constraints: tuple[Callable[[Schedule], bool], ...]) -> Callable[[Schedule], bool]:
    """Check hard constraints of a schedule."""
    return lambda s: not any(constraint(s) for constraint in constraints)


def generate_stable_config_hash(config: TournamentConfig, model: FitnessModel) -> int:
    """Generate a stable hash for a given tournament configuration."""
    representation = (
        config.canonical_round_tuples,
        config.canonical_roundreqs_tuple,
        model.penalties.minbreak_target,
        model.penalties.minbreak,
        model.penalties.zeros,
        config.num_teams,
    )
    # Using hashlib over built-in hash for stability
    return int(hashlib.sha256(str(representation).encode()).hexdigest(), 16)


def build_ga_context(cfg: AppConfig) -> GaContext:
    """Build and return a GA context."""
    evt_repo = build_evt_repo(cfg.tournament.rounds)
    evt_prop = build_evt_prop(evt_repo.mapping)
    _run_preflight_checks(evt_prop, evt_repo)
    Schedule.ctx = ScheduleContext(
        conflict_map=evt_repo.conflict_map,
        event_props=evt_prop,
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
        opponents = FitnessBenchmarkOpponent(cfg.tournament, evt_repo).benchmark()
        best_timeslot_score = FitnessBenchmarkBreaktime(
            cfg.tournament,
            cfg.fitness.penalties.minbreak_target,
            cfg.fitness.penalties.minbreak,
            cfg.fitness.penalties.zeros,
        ).benchmark()
        save_fitness_benchmark(benchmark_path, BenchmarkSeedData(opponents, best_timeslot_score))

    points = calc_ref_points(len(FitnessObjective), cfg.genetic.parameters.population_size)
    ref_directions = ReferenceDirections(points.shape[0], points, calc_norm_sq_of_refs(points))

    return GaContext(
        event_repo=evt_repo,
        event_properties=evt_prop,
        builder=ScheduleBuilderRandom(evt_prop, cfg.rng, cfg.tournament.round_idx_to_tpr, evt_repo.roundtypes),
        repairer=Repairer(cfg.tournament, evt_prop, cfg.rng),
        evaluator=FitnessEvaluator(
            cfg.tournament,
            evt_prop,
            opponents,
            best_timeslot_score,
            loc_weight_rounds_inter=cfg.fitness.location_weights.get_weights_tuple()[0],
            loc_weight_rounds_intra=cfg.fitness.location_weights.get_weights_tuple()[1],
            agg_weights=cfg.fitness.aggregation.get_weights_tuple(),
            min_fitness_weight=cfg.fitness.aggregation.min_fit,
            obj_weights=np.array(cfg.fitness.objectives.get_weights_tuple(), dtype=float),
            minbreak_target=cfg.fitness.penalties.minbreak_target,
            minbreak_penalty=cfg.fitness.penalties.minbreak,
            zeros_penalty=cfg.fitness.penalties.zeros,
        ),
        checker=checker,
        nsga3=NSGA3(cfg.rng, ref_directions, NonDominatedSorting()),
        selection=RandomSelect(cfg.rng),
        crossovers=build_crossovers(
            cfg.rng, cfg.genetic.operator.crossover.types, cfg.genetic.operator.crossover.k_vals, evt_repo, evt_prop
        ),
        mutations=build_mutations(cfg.rng, cfg.genetic.operator.mutation.types, evt_repo, evt_prop),
    )


def generate_events_from_round(r: TournamentRound, event_idx_iter: Iterator[int]) -> Iterator[Event]:
    """Generate all possible Events for a given TournamentRound configuration.

    Args:
        r (TournamentRound): The tournament round configuration to generate events for.
        event_idx_iter (Iterator[int]): An iterator to generate unique event IDs.

    Yields:
        Event: An event for the round with a time slot and a location.

    """
    for ts in r.timeslots:
        if r.teams_per_round == 1:
            for loc in r.locations:
                event = Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
                yield event
        elif r.teams_per_round == 2:
            event1 = Event()
            for loc in r.locations:
                if loc.side == 1:
                    event1 = Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
                elif loc.side == 2:
                    event2 = Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
                    event1.pair(event2)
                    yield from (event1, event2)


def build_evt_repo(rounds: tuple[TournamentRound, ...]) -> EventRepository:
    """Build an EventRepository from the tournament configuration."""
    event_iter = itertools.count()
    events = tuple(e for r in rounds for e in generate_events_from_round(r, event_iter))
    singles_or_side1 = tuple(e for e in events if e.paired == -1 or (e.paired != -1 and e.location.side == 1))
    timeslots = defaultdict(list)
    matches = defaultdict(list)
    roundtypes = defaultdict(list)
    for e in events:
        timeslots[(e.roundtype_idx, e.timeslot.idx)].append(e.idx)
        roundtypes[e.roundtype_idx].append(e.idx)
    for e in singles_or_side1:
        if e.paired != -1:
            matches[e.roundtype_idx].append((e.idx, e.paired))
    n = len(events)
    c_matrix = np.full((n, n), fill_value=False, dtype=bool)
    for e1, e2 in itertools.combinations(events, 2):
        if e1.timeslot.overlaps(e2.timeslot):
            e1.conflicts.append(e2.idx)
            e2.conflicts.append(e1.idx)
            c_matrix[e1.idx, e2.idx] = True
            c_matrix[e2.idx, e1.idx] = True
    for e in events:
        e.conflicts = sorted(set(e.conflicts))
        logger.debug("%s has %d conflicts: %s", e, len(e.conflicts), e.conflicts)
    for i in range(n):
        c_matrix[i, i] = True  # An event conflicts with itself
    return EventRepository(
        events=events,
        events_idx=np.array([e.idx for e in events], dtype=int),
        singles_or_side1_idx=np.array([e.idx for e in singles_or_side1], dtype=int),
        conflict_map={e.idx: set(e.conflicts) for e in events},
        mapping={e.idx: e for e in events},
        roundtypes=roundtypes,
        timeslots=timeslots,
        matches=matches,
    )


def build_evt_prop(event_map: dict[int, Event]) -> EventProperties:
    """Build EventProperties from an event mapping."""
    ep_dtype = np.dtype(
        [
            ("roundtype", "U50"),
            ("roundtype_idx", int),
            ("timeslot", object),
            ("timeslot_idx", int),
            ("start", int),
            ("stop_active", int),
            ("stop_cycle", int),
            ("location", object),
            ("loc_str", "U50"),
            ("loc_type", "U50"),
            ("loc_idx", int),
            ("loc_name", int),
            ("loc_side", int),
            ("teams_per_round", int),
            ("paired_idx", int),
        ]
    )
    ep: np.ndarray = np.zeros(len(event_map), dtype=ep_dtype)
    for i, e in event_map.items():
        ep[i]["roundtype"] = e.roundtype
        ep[i]["roundtype_idx"] = e.roundtype_idx
        ep[i]["timeslot"] = e.timeslot
        ep[i]["timeslot_idx"] = e.timeslot.idx
        ep[i]["start"] = int(e.timeslot.start.timestamp())
        ep[i]["stop_active"] = int(e.timeslot.stop_active.timestamp())
        ep[i]["stop_cycle"] = int(e.timeslot.stop_cycle.timestamp())
        ep[i]["location"] = e.location
        ep[i]["loc_str"] = str(e.location)
        ep[i]["loc_type"] = e.location.locationtype
        ep[i]["loc_idx"] = e.location.idx
        ep[i]["loc_name"] = e.location.name
        ep[i]["loc_side"] = e.location.side
        ep[i]["teams_per_round"] = e.location.teams_per_round
        ep[i]["paired_idx"] = e.paired
    return EventProperties(
        roundtype=ep["roundtype"],
        roundtype_idx=ep["roundtype_idx"],
        timeslot=ep["timeslot"],
        timeslot_idx=ep["timeslot_idx"],
        start=ep["start"],
        stop_active=ep["stop_active"],
        stop_cycle=ep["stop_cycle"],
        location=ep["location"],
        loc_str=ep["loc_str"],
        loc_type=ep["loc_type"],
        loc_idx=ep["loc_idx"],
        loc_name=ep["loc_name"],
        loc_side=ep["loc_side"],
        teams_per_round=ep["teams_per_round"],
        paired_idx=ep["paired_idx"],
    )


def run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    app_cfg_model = build_app_config_model(config_path)
    cfg = build_app_config(app_cfg_model)
    ctx = build_ga_context(cfg)
    _import_schedule(cfg, ctx)
    seed_file = Path(cfg.runtime.seed_file).resolve()
    seed_pop = load_ga(seed_file, cfg.tournament)
    ga = GA(
        context=ctx,
        genetic_model=cfg.genetic,
        rng=cfg.rng,
        observers=_get_observers(progress, task_id),
        operator_stats=OperatorStats(Counter(), _get_tracker(ctx.crossovers), _get_tracker(ctx.mutations)),
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
            cfg.rng,
            len(seed_pop),
            cfg.io.imports.seed_island_strategy,
            cfg.io.imports.seed_pop_sort,
            cfg.genetic.parameters.num_islands,
            cfg.genetic.parameters.population_size,
        ).get_island_seed_map(),
    )
    ga.run()
    _finalize_ga_results(cfg, ctx, seed_file, ga)
    return ga
