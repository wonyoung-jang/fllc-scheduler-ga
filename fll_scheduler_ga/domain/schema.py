"""Pydantic models for application configuration."""

import datetime as dt
import itertools
import logging
import math
import pprint as pp
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from fll_scheduler_ga.constants import (
    CMAP_NAME_DEFAULT,
    CONFIG_FILE_DEFAULT,
    OUTPUT_DIR_DEFAULT,
    PICKLE_FILE_SCHEDULES,
    RANDOM_SEED_RANGE,
    CrossoverOp,
    MutationOp,
    SeedIslandStrategy,
    SeedPopSort,
)
from fll_scheduler_ga.domain.model import DEFAULT_DT, Location, TimeSlot, TournamentConfig, TournamentRound

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

TIME_FORMAT_MAP = {12: "%I:%M %p", 24: "%H:%M"}


### GeneticModel
class GaParameterModel(BaseModel):
    """Genetic Algorithm parameters."""

    population_size: int = Field(default=2, ge=2)
    generations: int = Field(default=128, ge=1)
    offspring_size: int = Field(default=1, ge=1)
    crossover_chance: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_chance: float = Field(default=0.4, ge=0.0, le=1.0)
    num_islands: int = Field(default=1, ge=1)
    migration_interval: int = Field(default=10, ge=1)
    migration_size: int = Field(default=1, ge=0)


class CrossoverModel(BaseModel):
    """Configuration for crossover operators."""

    types: tuple[CrossoverOp | str, ...] = ()
    k_vals: tuple[int, ...] = ()


class MutationModel(BaseModel):
    """Configuration for mutation operators."""

    types: tuple[MutationOp | str, ...] = ()


class OperatorModel(BaseModel):
    """Container for operator configurations."""

    crossover: CrossoverModel = Field(default_factory=CrossoverModel)
    mutation: MutationModel = Field(default_factory=MutationModel)


class StagnationModel(BaseModel):
    """Configuration for stagnation handling."""

    enable: bool = False
    proportion: float = Field(default=0.8, ge=0.0, le=1.0)
    threshold: int = 20
    cooldown: int = 50


class GeneticModel(BaseModel):
    """Configuration for the genetic algorithm."""

    rng_seed: int | str | None = None
    parameters: GaParameterModel = Field(default_factory=GaParameterModel)
    operator: OperatorModel = Field(default_factory=OperatorModel)
    stagnation: StagnationModel = Field(default_factory=StagnationModel)


class RuntimeModel(BaseModel):
    """Configuration for command-line arguments and runtime flags."""

    add_import_to_population: bool = True
    flush: bool = False
    flush_benchmarks: bool = False
    import_file: str = ""
    seed_file: str = Field(default=PICKLE_FILE_SCHEDULES, min_length=1)


### IOModel
class ImportModel(BaseModel):
    """Configuration for import options."""

    seed_pop_sort: str = SeedPopSort.RANDOM
    seed_island_strategy: str = SeedIslandStrategy.DISTRIBUTED

    @model_validator(mode="after")
    def validate(self) -> ImportModel:
        """Validate import options."""
        if self.seed_pop_sort not in tuple(SeedPopSort):
            msg = f"Invalid seed_pop_sort: {self.seed_pop_sort}. Must be one of {[e.value for e in SeedPopSort]}."
            raise ValueError(msg)
        if self.seed_island_strategy not in tuple(SeedIslandStrategy):
            msg = (
                f"Invalid seed_island_strategy: {self.seed_island_strategy}. "
                f"Must be one of {[e.value for e in SeedIslandStrategy]}."
            )
            raise ValueError(msg)
        return self


class ExportModel(BaseModel):
    """Configuration for export options."""

    output_dir: str = Field(default=OUTPUT_DIR_DEFAULT, min_length=1)
    summary_reports: bool = True
    schedules_csv: bool = True
    schedules_html: bool = True
    schedules_team_csv: bool = True
    pareto_summary: bool = True
    plot_fitness: bool = True
    plot_parallel: bool = True
    plot_scatter: bool = True
    front_only: bool = True
    no_plotting: bool = False
    cmap_name: str = Field(default=CMAP_NAME_DEFAULT, min_length=1)
    team_identities: dict[int, str] = Field(default_factory=dict)


class IOModel(BaseModel):
    """Configuration for input/output options."""

    imports: ImportModel = Field(default_factory=ImportModel)
    exports: ExportModel = Field(default_factory=ExportModel)


### FitnessModel
class AggregationWeightsModel(BaseModel):
    """Configuration for aggregation weights."""

    mean: float | int = Field(default=1.0, ge=0.0)
    variation: float | int = Field(default=1.0, ge=0.0)
    range: float | int = Field(default=1.0, ge=0.0)
    min_fit: float = Field(default=0.3, ge=0.0)

    def get_weights_tuple(self) -> tuple[float, ...]:
        """Return the aggregation weights as a tuple."""
        weights = (self.mean, self.variation, self.range)
        total = sum(weights)
        return tuple(w / total for w in weights)


class ObjectiveWeightsModel(BaseModel):
    """Configuration for fitness objective weights."""

    breaktime: float | int = Field(default=1.0, ge=0.0)
    opponents: float | int = Field(default=1.0, ge=0.0)
    locations: float | int = Field(default=1.0, ge=0.0)

    def get_weights_tuple(self) -> tuple[float, ...]:
        """Return the objective weights as a tuple."""
        weights = (self.breaktime, self.locations, self.opponents)
        maximum = max(*weights)
        return tuple(w / maximum for w in weights)


class LocationWeightsModel(BaseModel):
    """Configuration for location weights."""

    inter_rounds: float = Field(default=0.5, ge=0.0)
    intra_rounds: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def validate(self) -> LocationWeightsModel:
        """Validate that weights sum to 1.0."""
        total = self.inter_rounds + self.intra_rounds
        if total <= 0.0:
            self.inter_rounds = 0.5
            self.intra_rounds = 0.5
            logger.warning("Location weights sum to zero: resetting to equal weights of 0.5 each.")
        return self

    def get_weights_tuple(self) -> tuple[float, float]:
        """Return the location weights as a tuple."""
        total = self.inter_rounds + self.intra_rounds
        return (self.inter_rounds / total, self.intra_rounds / total)


class PenaltyModel(BaseModel):
    """Configuration for penalty weights."""

    zeros: float = Field(default=0.0001, lt=1.0, ge=0.0)
    minbreak: float = Field(default=0.3, lt=1.0, ge=0.0)
    minbreak_target: int = Field(default=30, gt=0)


class FitnessModel(BaseModel):
    """Configuration for fitness weights."""

    aggregation: AggregationWeightsModel = Field(default_factory=AggregationWeightsModel)
    objectives: ObjectiveWeightsModel = Field(default_factory=ObjectiveWeightsModel)
    location_weights: LocationWeightsModel = Field(default_factory=LocationWeightsModel)
    penalties: PenaltyModel = Field(default_factory=PenaltyModel)


### TournamentModel
class LocationModel(BaseModel):
    """Input model for a location type."""

    name: str = Field(default="", min_length=1)
    count: int = Field(default=1, ge=1)
    sides: int = Field(default=1, ge=1)


class RoundModel(BaseModel):
    """Input model for a tournament round."""

    roundtype: str = Field(default="", min_length=1)
    location: str = Field(default="", min_length=1)
    rounds_per_team: int = Field(default=1, ge=1)
    teams_per_round: int = Field(default=1, ge=1)
    start_time: str = ""
    stop_time: str = ""
    times: list[str] = Field(default_factory=list)
    duration_cycle: int = Field(default=0, ge=0)
    duration_active: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate(self) -> RoundModel:
        """Validate that rounds_per_team and teams_per_round are positive."""
        if self.stop_time and not self.start_time:
            msg = f"Round '{self.roundtype}' has stop_time defined but no start_time."
            raise ValueError(msg)
        if not (self.start_time or self.times):
            msg = f"Round '{self.roundtype}' must have either start_time or times defined."
            raise ValueError(msg)
        if self.duration_active > self.duration_cycle:
            msg = f"Round '{self.roundtype}' has duration_active greater than duration_cycle."
            raise ValueError(msg)
        return self


class TournamentModel(BaseModel):
    """Root model for tournament configuration from JSON."""

    teams: int | tuple[int | str, ...] = 1
    locations: tuple[LocationModel, ...] = ()
    rounds: tuple[RoundModel, ...] = ()


### AppConfigModel
class AppConfigModel(BaseModel):
    """Root model for the entire application configuration from JSON."""

    genetic: GeneticModel = Field(default_factory=GeneticModel)
    runtime: RuntimeModel = Field(default_factory=RuntimeModel)
    io: IOModel = Field(default_factory=IOModel)
    fitness: FitnessModel = Field(default_factory=FitnessModel)
    tournament: TournamentModel = Field(default_factory=TournamentModel)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    genetic: GeneticModel
    runtime: RuntimeModel
    io: IOModel
    fitness: FitnessModel
    tournament: TournamentConfig
    rng: np.random.Generator


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


def _calc_num_timeslots(n_times: int, n_locs: int, n_teams: int, rounds_per_team: int) -> int:
    """Calculate the number of timeslots needed for a round."""
    if n_times > 0:
        return n_times
    if n_locs > 0:
        return math.ceil((n_teams * rounds_per_team) / n_locs)
    msg = "Cannot calculate number of timeslots without times or locations."
    raise ValueError(msg)


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
    if n_timeslots <= 0:
        msg = "n_timeslots must be greater than zero to validate duration."
        raise ValueError(msg)
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
        return
    current = start_dt
    for _ in range(n_timeslots):
        stop_cycle = current + dur_cycle
        stop_active = current + dur_active
        yield (current, stop_active, stop_cycle)
        current = stop_cycle


def _get_all_sorted_attr(objects: Iterable[Any], get_by: str, sort_by: str) -> tuple[Any, ...]:
    """Get all attributes of the TournamentRound objects."""
    return tuple(
        sorted(itertools.chain.from_iterable(getattr(r, get_by) for r in objects), key=lambda x: getattr(x, sort_by))
    )


def _get_team_identities(teams: tuple[str, ...]) -> dict[int, str]:
    """Return a mapping of team indices to team identities."""
    return dict(enumerate(teams, start=1))


def _get_teams_list(teams: tuple[int | str, ...] | int) -> tuple[str, ...]:
    """Return a tuple of team identifiers."""
    if isinstance(teams, int):
        return tuple(str(i) for i in range(1, teams + 1))
    return tuple(str(t) for t in teams)


def _get_rng_seed(seed: int | str | None) -> int:
    """Return the RNG seed as an integer."""
    if isinstance(seed, int):
        return seed
    return int(
        np.random.default_rng().integers(*RANDOM_SEED_RANGE)
        if seed is None
        else abs(hash(seed)) % (RANDOM_SEED_RANGE[1] + 1)
    )


def _parse_locations(models: tuple[LocationModel, ...]) -> tuple[Location, ...]:
    """Parse location models into Location instances."""
    _idx_counter = itertools.count()

    def _generate_locations() -> Iterator[Location]:
        for loctype in models:
            for name in range(1, loctype.count + 1):
                for side_iter in range(1, loctype.sides + 1):
                    yield Location(
                        idx=next(_idx_counter),
                        locationtype=loctype.name,
                        name=name,
                        side=-1 if loctype.sides == 1 else side_iter,
                        teams_per_round=loctype.sides,
                    )

    locations = tuple(_generate_locations())
    if not locations:
        msg = "No locations defined in the configuration file."
        raise ValueError(msg)
    return locations


def _get_time_fmt(round_models: Iterable[RoundModel]) -> str:
    """Get the time format from the rounds configuration."""

    def _generate_all_time_strs() -> Iterator[str]:
        for rm in round_models:
            if rm.start_time:
                yield rm.start_time
            if rm.stop_time:
                yield rm.stop_time
            yield from rm.times

    format_counts = Counter(_infer_time_format(t) for t in _generate_all_time_strs() if t)
    if not format_counts:
        msg = "No time strings found in configuration to infer time format."
        raise ValueError(msg)
    if len(format_counts) != 1:
        msg = "Conflicting time formats found in configuration times."
        raise ValueError(msg)
    return str(format_counts.most_common(1)[0][0])


def _are_rounds_overlapping(rounds: Iterable[TournamentRound]) -> bool:
    """Check if any rounds are interleaved in time."""
    _starts = (r.start_time for r in rounds)
    _stops = (r.stop_time for r in rounds)
    timeslots = tuple(
        TimeSlot(idx=0, start=start, stop_active=stop_cycle, stop_cycle=stop_cycle)
        for start, stop_cycle in zip(_starts, _stops, strict=True)
    )
    return any(timeslots[i].overlaps(timeslots[j]) for i in range(len(timeslots)) for j in range(i + 1, len(timeslots)))


def _load_tournament_config(
    n_teams: int, round_models: tuple[RoundModel, ...], locations: tuple[Location, ...]
) -> TournamentConfig:
    """Load and return the tournament configuration from the validated model."""
    time_fmt = _get_time_fmt(round_models)
    TimeSlot.time_fmt = time_fmt
    rounds = parse_rounds_config(round_models, n_teams, time_fmt, locations)
    if not rounds:
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)
    roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
    round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}
    total_slots_required = sum(r.slots_required for r in rounds)
    unique_opponents_possible = 1 <= max(roundreqs.values()) <= n_teams - 1
    max_events_per_team = sum(roundreqs.values())
    all_locations = _get_all_sorted_attr(rounds, get_by="locations", sort_by="idx")
    all_timeslots = _get_all_sorted_attr(rounds, get_by="timeslots", sort_by="idx")
    is_interleaved = _are_rounds_overlapping(rounds)
    return TournamentConfig(
        num_teams=n_teams,
        time_fmt=time_fmt,
        rounds=rounds,
        roundreqs=roundreqs,
        round_idx_to_tpr=round_idx_to_tpr,
        total_slots_required=total_slots_required,
        unique_opponents_possible=unique_opponents_possible,
        max_events_per_team=max_events_per_team,
        all_locations=all_locations,
        all_timeslots=all_timeslots,
        is_interleaved=is_interleaved,
    )


def parse_rounds_config(
    models: tuple[RoundModel, ...], n_teams: int, time_fmt: str, all_locations: tuple[Location, ...]
) -> tuple[TournamentRound, ...]:
    """Parse and return TournamentRound objects from the configuration."""

    def _generate_rounds(timeslot_idx_iter: Iterator[int]) -> Iterator[TournamentRound]:
        for roundtype_idx, rm in enumerate(models):
            _times = rm.times
            _rounds_per_team = rm.rounds_per_team
            _location = rm.location
            locations = tuple(loc for loc in all_locations if loc.locationtype == _location)
            _n_locations = len(locations)
            start_dt = _parse_time_str(rm.start_time, time_fmt)
            stop_dt = _parse_time_str(rm.stop_time, time_fmt)
            times_dt = tuple(_parse_time_str(t, time_fmt) for t in _times) if _times else ()
            _n_timeslots = _calc_num_timeslots(len(times_dt), _n_locations, n_teams, _rounds_per_team)
            start_stop = (start_dt, stop_dt)
            dur_tdelta_cycle = _validate_duration(start_stop, times_dt, rm.duration_cycle, _n_timeslots)
            dur_tdelta_active = _validate_duration(start_stop, times_dt, rm.duration_active, _n_timeslots)
            timeslots_iter = _init_timeslots(times_dt, dur_tdelta_cycle, dur_tdelta_active, _n_timeslots, start_dt)
            timeslots = tuple(
                TimeSlot(idx=next(timeslot_idx_iter), start=start, stop_active=stop_active, stop_cycle=stop_cycle)
                for start, stop_active, stop_cycle in timeslots_iter
            )
            round_start_time = timeslots[0].start
            round_stop_time = timeslots[-1].stop_cycle
            times_dt = tuple(ts.start for ts in timeslots)
            slots_total = _n_timeslots * _n_locations
            slots_required = n_teams * _rounds_per_team
            slots_empty = slots_total - slots_required
            if slots_empty < 0:
                msg = (
                    "Insufficient capacity for TournamentRound (required > available).\n"
                    "Suggestion: increase number of locations or timeslots."
                )
                raise ValueError(msg)
            unfilled_allowed = slots_empty > 0
            yield TournamentRound(
                roundtype=rm.roundtype,
                roundtype_idx=roundtype_idx,
                rounds_per_team=_rounds_per_team,
                teams_per_round=rm.teams_per_round,
                times=times_dt,
                start_time=round_start_time,
                stop_time=round_stop_time,
                duration_minutes=dur_tdelta_cycle,
                location_type=_location,
                locations=locations,
                num_timeslots=_n_timeslots,
                timeslots=timeslots,
                slots_total=slots_total,
                slots_required=slots_required,
                slots_empty=slots_empty,
                unfilled_allowed=unfilled_allowed,
            )

    timeslot_idx_iter = itertools.count()
    return tuple(sorted(_generate_rounds(timeslot_idx_iter), key=lambda r: r.start_time))


def log_appconfig_creation_info(cfg: AppConfig) -> None:
    """Log information about the application configuration creation."""
    logger.debug("Initialized AppConfig: %s", pp.pformat(cfg))
    for r in cfg.tournament.rounds:
        logger.debug("Initialized tournament round: %s", pp.pformat(r))
    logger.debug("Initialized tournament configuration: %s", pp.pformat(cfg.tournament))
    logger.debug("Initialized operator configuration: %s", pp.pformat(cfg.genetic.operator))
    logger.debug("Initialized genetic algorithm parameters: %s", pp.pformat(cfg.genetic.parameters))


def build_app_config(path: Path = CONFIG_FILE_DEFAULT) -> AppConfig:
    """Create and return the application configuration."""
    if not path.exists():
        msg = f"Configuration file does not exist at: {path}"
        raise FileNotFoundError(msg)
    model = AppConfigModel.model_validate_json(path.read_text())
    teams_list = _get_teams_list(model.tournament.teams)
    model.io.exports.team_identities = _get_team_identities(teams_list)
    locations = _parse_locations(model.tournament.locations)
    tournament_config = _load_tournament_config(len(teams_list), model.tournament.rounds, locations)
    rng = np.random.default_rng(_get_rng_seed(model.genetic.rng_seed))
    return AppConfig(
        genetic=model.genetic,
        runtime=model.runtime,
        io=model.io,
        fitness=model.fitness,
        tournament=tournament_config,
        rng=rng,
    )
