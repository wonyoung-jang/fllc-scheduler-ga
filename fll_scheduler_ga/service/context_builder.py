"""Context builder for the genetic algorithm."""

import hashlib
import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.adapter.seeder import load_fitness_benchmark, save_fitness_benchmark
from fll_scheduler_ga.constants import BENCHMARKS_CACHE, FitnessObjective
from fll_scheduler_ga.domain.model import (
    BenchmarkSeedData,
    Event,
    EventProperties,
    EventRepository,
    Schedule,
    ScheduleContext,
    TimeSlot,
    TournamentConfig,
    TournamentRound,
)
from fll_scheduler_ga.genetic.context import GaContext, ScheduleBuilderRandom
from fll_scheduler_ga.genetic.fitness import FitnessBenchmarkBreaktime, FitnessBenchmarkOpponent, FitnessEvaluator
from fll_scheduler_ga.genetic.operator import (
    NSGA3,
    RandomSelect,
    Repairer,
    build_crossovers,
    build_mutations,
    calc_norm_sq_of_refs,
    calc_ref_points,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fll_scheduler_ga.adapter.schema import AppConfig, FitnessModel


logger = logging.getLogger(__name__)


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
                yield Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
        elif r.teams_per_round == 2:
            e1 = Event()
            for loc in r.locations:
                if loc.side == 1:
                    e1 = Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
                elif loc.side == 2:
                    e2 = Event(next(event_idx_iter), r.roundtype, r.roundtype_idx, ts, loc)
                    e1.pair(e2)
                    yield from (e1, e2)


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


def preflight(prop: EventProperties, repo: EventRepository) -> None:
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
    preflight(evt_prop, evt_repo)
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
        nsga3=NSGA3(cfg.rng, points.shape[0], points, calc_norm_sq_of_refs(points)),
        selection=RandomSelect(cfg.rng),
        crossovers=build_crossovers(
            cfg.rng, cfg.genetic.operator.crossover.types, cfg.genetic.operator.crossover.k_vals, evt_repo, evt_prop
        ),
        mutations=build_mutations(cfg.rng, cfg.genetic.operator.mutation.types, evt_repo, evt_prop),
    )
