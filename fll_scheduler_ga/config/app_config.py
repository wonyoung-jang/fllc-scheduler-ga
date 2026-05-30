"""Configuration for the FLL Scheduler GA application."""

import datetime as dt
import itertools
import logging
import math
import pprint as pp
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.config.schemas import (
    AppConfigModel,
    FitnessModel,
    GeneticModel,
    IOModel,
    LocationModel,
    RoundModel,
    RuntimeModel,
)
from fll_scheduler_ga.constants import CONFIG_FILE_DEFAULT, RANDOM_SEED_RANGE
from fll_scheduler_ga.domain.model import DEFAULT_DT, Location, TimeSlot, TournamentConfig, TournamentRound

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

TIME_FORMAT_MAP = {12: "%I:%M %p", 24: "%H:%M"}


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    genetic: GeneticModel
    runtime: RuntimeModel
    io: IOModel
    fitness: FitnessModel
    tournament: TournamentConfig
    rng: np.random.Generator


def parse_time_str(dt_str: str, fmt: str) -> dt.datetime:
    """Parse a time string into a datetime object."""
    if not dt_str:
        return DEFAULT_DT
    return dt.datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=dt.UTC)


def infer_time_format(dt_str: str) -> str | None:
    """Infer the time format from a sample time string."""
    for fmt in TIME_FORMAT_MAP.values():
        try:
            parse_time_str(dt_str, fmt)
        except ValueError:
            continue
        return fmt
    return None


def calc_num_timeslots(n_times: int, n_locs: int, n_teams: int, rounds_per_team: int) -> int:
    """Calculate the number of timeslots needed for a round."""
    if n_times > 0:
        return n_times
    if n_locs > 0:
        return math.ceil((n_teams * rounds_per_team) / n_locs)
    msg = "Cannot calculate number of timeslots without times or locations."
    raise ValueError(msg)


def validate_duration(
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


def init_timeslots(
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

    format_counts = Counter(infer_time_format(t) for t in _generate_all_time_strs() if t)
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
            start_dt = parse_time_str(rm.start_time, time_fmt)
            stop_dt = parse_time_str(rm.stop_time, time_fmt)
            times_dt = tuple(parse_time_str(t, time_fmt) for t in _times) if _times else ()
            _n_timeslots = calc_num_timeslots(len(times_dt), _n_locations, n_teams, _rounds_per_team)
            start_stop = (start_dt, stop_dt)
            dur_tdelta_cycle = validate_duration(start_stop, times_dt, rm.duration_cycle, _n_timeslots)
            dur_tdelta_active = validate_duration(start_stop, times_dt, rm.duration_active, _n_timeslots)
            timeslots = tuple(
                TimeSlot(idx=next(timeslot_idx_iter), start=start, stop_active=stop_active, stop_cycle=stop_cycle)
                for start, stop_active, stop_cycle in init_timeslots(
                    times_dt, dur_tdelta_cycle, dur_tdelta_active, _n_timeslots, start_dt
                )
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
