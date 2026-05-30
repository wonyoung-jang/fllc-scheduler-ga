"""Configuration for the FLL Scheduler GA application."""

import itertools
import logging
import pprint as pp
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fll_scheduler_ga.config.pydantic_schemas import (
    AppConfigModel,
    FitnessModel,
    GeneticModel,
    IOModel,
    LocationModel,
    RoundModel,
    RuntimeModel,
)
from fll_scheduler_ga.constants import CONFIG_FILE_DEFAULT, RANDOM_SEED_RANGE
from fll_scheduler_ga.domain.location import Location
from fll_scheduler_ga.domain.model import TournamentConfig, TournamentRound, are_rounds_overlapping
from fll_scheduler_ga.domain.timeslot import (
    TimeSlot,
    calc_num_timeslots,
    infer_time_format,
    init_timeslots,
    parse_time_str,
    validate_duration,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


def get_all_sorted_attr(objects: Iterable[Any], get_by: str, sort_by: str) -> tuple[Any, ...]:
    """Get all attributes of the TournamentRound objects."""
    all_of = sorted(
        itertools.chain.from_iterable(getattr(r, get_by) for r in objects), key=lambda x: getattr(x, sort_by)
    )
    return tuple(all_of)


def get_team_identities(teams: tuple[str, ...]) -> dict[int, str]:
    """Return a mapping of team indices to team identities."""
    return dict(enumerate(teams, start=1))


def get_teams_list(teams: tuple[int | str, ...] | int) -> tuple[str, ...]:
    """Return a tuple of team identifiers."""
    if isinstance(teams, int):
        return tuple(str(i) for i in range(1, teams + 1))
    return tuple(str(t) for t in teams)


def get_rng_seed(seed: int | str | None) -> int:
    """Return the RNG seed as an integer."""
    if isinstance(seed, int):
        return seed
    return int(
        np.random.default_rng().integers(*RANDOM_SEED_RANGE)
        if seed is None
        else abs(hash(seed)) % (RANDOM_SEED_RANGE[1] + 1)
    )


def parse_locations(models: tuple[LocationModel, ...]) -> tuple[Location, ...]:
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


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    genetic: GeneticModel
    runtime: RuntimeModel
    io: IOModel
    fitness: FitnessModel
    tournament: TournamentConfig
    rng: np.random.Generator

    @classmethod
    def build(cls, path: Path | None = None) -> AppConfig:
        """Create and return the application configuration."""
        if path is None:
            path = CONFIG_FILE_DEFAULT
        if not path.exists():
            msg = f"Configuration file does not exist at: {path}"
            raise FileNotFoundError(msg)
        config_data = path.read_text()
        config_model = AppConfigModel.model_validate_json(config_data)
        return cls.build_from_model(config_model)

    @classmethod
    def build_from_model(cls, model: AppConfigModel) -> AppConfig:
        """Create and return the application configuration from a Pydantic model."""
        teams_list = get_teams_list(model.tournament.teams)
        model.io.exports.team_identities = get_team_identities(teams_list)
        n_teams = len(teams_list)
        _location_models = model.tournament.locations
        locations = parse_locations(_location_models)
        tournament_config = cls.load_tournament_config(n_teams, model.tournament.rounds, locations)
        seed = get_rng_seed(model.genetic.rng_seed)
        rng = np.random.default_rng(seed)
        return AppConfig(
            genetic=model.genetic,
            runtime=model.runtime,
            io=model.io,
            fitness=model.fitness,
            tournament=tournament_config,
            rng=rng,
        )

    @classmethod
    def load_tournament_config(
        cls, n_teams: int, round_models: tuple[RoundModel, ...], locations: tuple[Location, ...]
    ) -> TournamentConfig:
        """Load and return the tournament configuration from the validated model."""
        time_fmt = cls.get_time_fmt(round_models)
        TimeSlot.time_fmt = time_fmt
        rounds = cls.parse_rounds_config(round_models, n_teams, time_fmt, locations)
        if not rounds:
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)
        roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}
        total_slots_required = sum(r.slots_required for r in rounds)
        unique_opponents_possible = 1 <= max(roundreqs.values()) <= n_teams - 1
        max_events_per_team = sum(roundreqs.values())
        all_locations = get_all_sorted_attr(rounds, get_by="locations", sort_by="idx")
        all_timeslots = get_all_sorted_attr(rounds, get_by="timeslots", sort_by="idx")
        is_interleaved = are_rounds_overlapping(rounds)
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

    @classmethod
    def get_time_fmt(cls, round_models: Iterable[RoundModel]) -> str:
        """Get the time format from the rounds configuration."""

        def _generate_all_time_strs() -> Iterator[str]:
            for rm in round_models:
                if rm.start_time:
                    yield rm.start_time
                if rm.stop_time:
                    yield rm.stop_time
                yield from rm.times

        format_counts = Counter(infer_time_format(t) for t in _generate_all_time_strs() if t)
        if len(format_counts) != 1:
            msg = "Conflicting time formats found in configuration times."
            raise ValueError(msg)
        return str(format_counts.most_common(1)[0][0])

    @classmethod
    def parse_rounds_config(
        cls, round_models: tuple[RoundModel, ...], n_teams: int, time_fmt: str, all_locations: tuple[Location, ...]
    ) -> tuple[TournamentRound, ...]:
        """Parse and return TournamentRound objects from the configuration."""
        timeslot_idx_iter = itertools.count()

        def _generate_rounds() -> Iterator[TournamentRound]:
            for roundtype_idx, round_model in enumerate(round_models):
                _times = round_model.times
                _rounds_per_team = round_model.rounds_per_team
                _location = round_model.location
                locations = tuple(loc for loc in all_locations if loc.locationtype == _location)
                _n_locations = len(locations)
                start_dt = parse_time_str(round_model.start_time, time_fmt)
                stop_dt = parse_time_str(round_model.stop_time, time_fmt)
                times_dt = tuple(parse_time_str(t, time_fmt) for t in _times) if _times else ()
                _n_timeslots = calc_num_timeslots(len(times_dt), _n_locations, n_teams, _rounds_per_team)
                start_stop = (start_dt, stop_dt)
                dur_tdelta_cycle = validate_duration(start_stop, times_dt, round_model.duration_cycle, _n_timeslots)
                dur_tdelta_active = validate_duration(start_stop, times_dt, round_model.duration_active, _n_timeslots)
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
                    roundtype=round_model.roundtype,
                    roundtype_idx=roundtype_idx,
                    rounds_per_team=_rounds_per_team,
                    teams_per_round=round_model.teams_per_round,
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

        rounds = sorted(_generate_rounds(), key=lambda r: r.start_time)
        return tuple(rounds)

    def log_creation_info(self) -> None:
        """Log information about the application configuration creation."""
        logger.debug("Initialized AppConfig: %s", pp.pformat(self))
        for r in self.tournament.rounds:
            logger.debug("Initialized tournament round: %s", pp.pformat(r))
        logger.debug("Initialized tournament configuration: %s", pp.pformat(self.tournament))
        logger.debug("Initialized operator configuration: %s", pp.pformat(self.genetic.operator))
        logger.debug("Initialized genetic algorithm parameters: %s", pp.pformat(self.genetic.parameters))
