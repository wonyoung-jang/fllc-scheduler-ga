"""Module for building the application configuration from the validated model."""

import datetime as dt
import itertools
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.adapter.schema import AppConfig, AppConfigModel, LocationModel, RoundModel
from fll_scheduler_ga.constants import RANDOM_SEED_RANGE
from fll_scheduler_ga.domain.model import DEFAULT_DT, Location, TimeSlot, TournamentConfig, TournamentRound

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator


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


@dataclass(slots=True)
class AppConfigBuilder:
    """Builder for the application configuration from the validated model."""

    m: AppConfigModel

    def build(self) -> AppConfig:
        """Build and return the application configuration."""
        teams = parse_teams(self.m.tournament.teams)
        team_identities = dict(enumerate(teams, start=1))
        if not (locations := parse_locations(self.m.tournament.locations)):
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)
        time_fmt = parse_time_fmt(self.m.tournament.rounds)
        TimeSlot.fmt = time_fmt
        if not (rounds := parse_rounds(self.m.tournament.rounds, len(teams), time_fmt, locations)):
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)
        tournament_config = get_tournament_config(len(teams), time_fmt, rounds)
        rng = get_rng(self.m.genetic.rng_seed)
        return AppConfig(
            self.m.genetic,
            self.m.runtime,
            self.m.io,
            self.m.fitness,
            tournament_config,
            team_identities,
            rng,
            self.m.fitness.aggregation.min_fit,
            self.m.fitness.objectives.get_weights_tuple(),
            self.m.fitness.aggregation.get_weights_tuple(),
        )
