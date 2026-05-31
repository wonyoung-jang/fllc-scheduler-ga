"""Context for the genetic algorithm parts."""

from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import BENCHMARKS_CACHE
from fll_scheduler_ga.domain.model import EventFactory, EventProperties, build_event_props
from fll_scheduler_ga.domain.schedule import Schedule, ScheduleContext
from fll_scheduler_ga.fitness.benchmark import (
    FitnessBenchmark,
    FitnessBenchmarkBreaktime,
    FitnessBenchmarkOpponent,
    PickleBenchmarkRepository,
    generate_stable_config_hash,
)
from fll_scheduler_ga.fitness.evaluator import FitnessEvaluator
from fll_scheduler_ga.genetic.builder import ScheduleBuilderRandom
from fll_scheduler_ga.operators.crossover import build_crossovers
from fll_scheduler_ga.operators.mutation import build_mutations
from fll_scheduler_ga.operators.nsga3 import (
    NSGA3,
    NonDominatedSorting,
    ReferenceDirections,
    calc_norm_sq_of_refs,
    calc_ref_points,
)
from fll_scheduler_ga.operators.repairer import Repairer
from fll_scheduler_ga.operators.selection import RandomSelect

if TYPE_CHECKING:
    from collections.abc import Callable

    from fll_scheduler_ga.domain.model import TimeSlot, TournamentConfig
    from fll_scheduler_ga.domain.schema import AppConfig, ImportModel
    from fll_scheduler_ga.operators.crossover import Crossover
    from fll_scheduler_ga.operators.mutation import Mutation
    from fll_scheduler_ga.operators.selection import Selection

logger = getLogger(__name__)


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
    n_total_evts = cfg.tournament.get_n_total_events()
    evt_factory = EventFactory(config=cfg.tournament)
    evt_props = build_event_props(n_total_events=n_total_evts, event_map=evt_factory.mapping)
    _run_preflight_checks(evt_props, evt_factory)
    Schedule.ctx = ScheduleContext(
        conflict_map=evt_factory.conflict_map,
        event_props=evt_props,
        teams_list=np.arange(cfg.tournament.num_teams, dtype=int),
        teams_roundreqs_arr=np.tile(A=tuple(cfg.tournament.roundreqs.values()), reps=(cfg.tournament.num_teams, 1)),
        empty_schedule=np.full(n_total_evts, -1, dtype=int),
    )
    constraints = (
        lambda s: not s,
        lambda s: s.get_size() != cfg.tournament.total_slots_required,
        lambda s: s.any_rounds_needed(),
    )
    checker = _hard_constraint_checker(constraints)
    config_hash = generate_stable_config_hash(config=cfg.tournament, model=cfg.fitness)
    BENCHMARKS_CACHE.mkdir(parents=True, exist_ok=True)
    seed_file = BENCHMARKS_CACHE / f"benchmark_cache_{config_hash}.pkl"
    benchmark = FitnessBenchmark(
        config=cfg.tournament,
        model=cfg.fitness,
        repository=PickleBenchmarkRepository(path=seed_file),
        opponent_benchmarker=FitnessBenchmarkOpponent(config=cfg.tournament, event_factory=evt_factory),
        breaktime_benchmarker=FitnessBenchmarkBreaktime(
            config=cfg.tournament, event_factory=evt_factory, model=cfg.fitness
        ),
        flush_benchmarks=cfg.runtime.flush_benchmarks,
    )
    evaluator = FitnessEvaluator(
        config=cfg.tournament, event_properties=evt_props, benchmark=benchmark, model=cfg.fitness
    )
    points = calc_ref_points(evaluator.n_objectives, cfg.genetic.parameters.population_size)
    n_refs = points.shape[0]
    norm_sq = calc_norm_sq_of_refs(points)
    ref_directions = ReferenceDirections(n_refs=n_refs, points=points, norm_sq=norm_sq)
    crossovers = build_crossovers(cfg.rng, cfg.genetic.operator, evt_factory, evt_props)
    mutations = build_mutations(cfg.rng, cfg.genetic.operator, evt_factory, evt_props)
    return GaContext(
        app_config=cfg,
        event_factory=evt_factory,
        event_properties=evt_props,
        builder=ScheduleBuilderRandom(
            event_properties=evt_props,
            rng=cfg.rng,
            round_idx_to_tpr=cfg.tournament.round_idx_to_tpr,
            roundtype_events=evt_factory.roundtypes,
        ),
        repairer=Repairer(
            config=cfg.tournament, event_factory=evt_factory, event_properties=evt_props, rng=cfg.rng, checker=checker
        ),
        evaluator=evaluator,
        checker=checker,
        nsga3=NSGA3(rng=cfg.rng, refs=ref_directions, sorting=NonDominatedSorting()),
        selection=RandomSelect(cfg.rng),
        crossovers=crossovers,
        mutations=mutations,
    )


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    event_properties: EventProperties
    evaluator: FitnessEvaluator
    checker: Callable[[Schedule], bool]
    builder: ScheduleBuilderRandom
    repairer: Repairer
    nsga3: NSGA3
    selection: Selection
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]

    def build(self) -> Schedule:
        """Build a new schedule using the builder."""
        return self.builder.build()

    def check(self, schedule: Schedule) -> bool:
        """Check a schedule using the hard constraint checker."""
        return self.checker(schedule)

    def repair(self, schedule: Schedule) -> bool:
        """Repair a schedule using the repairer."""
        return self.repairer.repair(schedule)

    def evaluate(self, pop_array: np.ndarray) -> tuple[np.ndarray, ...]:
        """Evaluate a schedule using the fitness evaluator."""
        return self.evaluator.evaluate(pop_array)

    def select_parents(self, n: int, k: int = 2) -> np.ndarray:
        """Select parents using the selection operator."""
        return self.selection.select(n, k)

    def select_nsga3(self, fits: np.ndarray, n_select: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Select individuals using NSGA-III."""
        return self.nsga3.select(fits, n_select)

    @property
    def tournament_config(self) -> TournamentConfig:
        """Get the tournament configuration from the app config."""
        return self.app_config.tournament

    @property
    def import_model(self) -> ImportModel:
        """Get the imports model from the app config."""
        return self.app_config.io.imports

    @property
    def seed_island_strategy(self) -> str:
        """Get the seed island strategy from the app config."""
        return self.app_config.io.imports.seed_island_strategy
