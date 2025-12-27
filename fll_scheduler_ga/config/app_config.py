"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

import itertools
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data_model.location import Location
from ..data_model.timeslot import (
    TimeSlot,
    calc_num_timeslots,
    infer_time_format,
    init_timeslots,
    parse_time_str,
    validate_duration,
)
from .app_schemas import TournamentConfig, TournamentRound
from .constants import CONFIG_FILE_DEFAULT
from .schemas import (
    AppConfigModel,
    ExportModel,
    FitnessModel,
    GeneticModel,
    ImportModel,
    LocationModel,
    RoundModel,
    RuntimeModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    runtime: RuntimeModel
    imports: ImportModel
    exports: ExportModel
    genetic: GeneticModel
    fitness: FitnessModel
    tournament: TournamentConfig
    rng: np.random.Generator

    @classmethod
    def build(cls, path: Path | None = None) -> AppConfig:
        """Create and return the application configuration."""
        if path is None:
            path = CONFIG_FILE_DEFAULT.resolve()

        if not path.exists():
            msg = f"Configuration file does not exist at: {path}"
            raise FileNotFoundError(msg)

        with path.open(mode="r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_model = AppConfigModel.model_validate(config_data)
        return cls.build_from_model(config_model)

    @classmethod
    def build_from_model(cls, model: AppConfigModel) -> AppConfig:
        """Create and return the application configuration from a Pydantic model."""
        model.exports.team_identities = model.teams.get_team_ids()
        tournament_config = cls.load_tournament_config(model)
        rng_seed = model.genetic.get_rng_seed()
        rng = np.random.default_rng(rng_seed)
        return AppConfig(
            runtime=model.runtime,
            imports=model.imports,
            exports=model.exports,
            genetic=model.genetic,
            fitness=model.fitness,
            tournament=tournament_config,
            rng=rng,
        )

    @classmethod
    def load_tournament_config(cls, model: AppConfigModel) -> TournamentConfig:
        """Load and return the tournament configuration from the validated model."""
        n_teams = len(model.teams)
        round_models = model.rounds
        time_fmt = cls.get_time_fmt(round_models)
        TimeSlot.time_fmt = time_fmt

        all_locations = cls.parse_location_config(model.locations)
        if not all_locations:
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)

        rounds = cls.parse_rounds_config(round_models, n_teams, time_fmt, all_locations)
        if not rounds:
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}
        total_slots_required = sum(r.slots_required for r in rounds)
        unique_opponents_possible = 1 <= max(roundreqs.values()) <= n_teams - 1
        max_events_per_team = sum(roundreqs.values())

        all_locations = cls.get_all_attributes_of_rounds(rounds, round_attr="locations", sort_attr="idx")
        all_timeslots = cls.get_all_attributes_of_rounds(rounds, round_attr="timeslots", sort_attr="idx")
        is_interleaved = cls.check_interleaved(rounds)

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
    def get_time_fmt(cls, round_models: tuple[RoundModel, ...]) -> str:
        """Get the time format from the rounds configuration."""

        def _generate_all_time_strs() -> Iterator[str]:
            for rnd in round_models:
                if rnd.start_time:
                    yield rnd.start_time
                if rnd.stop_time:
                    yield rnd.stop_time
                yield from rnd.times

        format_counts = Counter(infer_time_format(t) for t in _generate_all_time_strs() if t)
        if len(format_counts) == 1:
            return str(format_counts.most_common(1)[0][0])

        msg = "Conflicting time formats found in configuration times."
        raise ValueError(msg)

    @classmethod
    def get_all_attributes_of_rounds(
        cls, rounds: tuple[TournamentRound, ...], round_attr: str, sort_attr: str
    ) -> tuple[Any, ...]:
        """Get all attributes of the TournamentRound objects."""
        all_of = list(itertools.chain.from_iterable(getattr(r, round_attr) for r in rounds))
        all_of.sort(key=lambda x: getattr(x, sort_attr))
        return tuple(all_of)

    @classmethod
    def check_interleaved(cls, rounds: tuple[TournamentRound, ...]) -> bool:
        """Check if any rounds are interleaved in time."""
        round_starts = (r.start_time for r in rounds)
        round_stops = (r.stop_time for r in rounds)
        timeslots = tuple(
            TimeSlot(
                idx=0,
                start=start,
                stop_active=stop_cycle,
                stop_cycle=stop_cycle,
            )
            for start, stop_cycle in zip(round_starts, round_stops, strict=True)
        )

        return any(
            timeslots[i].overlaps(timeslots[j]) for i in range(len(timeslots)) for j in range(i + 1, len(timeslots))
        )

    @classmethod
    def parse_rounds_config(
        cls,
        round_models: tuple[RoundModel, ...],
        n_teams: int,
        time_fmt: str,
        all_locations: tuple[Location, ...],
    ) -> tuple[TournamentRound, ...]:
        """Parse and return TournamentRound objects from the configuration."""
        timeslot_idx_iter = itertools.count()

        def _generate_rounds() -> Iterator[TournamentRound]:
            for roundtype_idx, rnd in enumerate(round_models):
                start_dt = parse_time_str(rnd.start_time, time_fmt)
                stop_dt = parse_time_str(rnd.stop_time, time_fmt)
                times_dt = tuple(parse_time_str(t, time_fmt) for t in rnd.times) if rnd.times else ()
                locations = tuple(loc for loc in all_locations if loc.locationtype == rnd.location)

                n_timeslots = calc_num_timeslots(len(times_dt), len(locations), n_teams, rnd.rounds_per_team)

                dur_tdelta_cycle = validate_duration(start_dt, stop_dt, times_dt, rnd.duration_cycle, n_timeslots)
                dur_tdelta_active = validate_duration(start_dt, stop_dt, times_dt, rnd.duration_active, n_timeslots)

                timeslots = tuple(
                    TimeSlot(
                        idx=next(timeslot_idx_iter),
                        start=start,
                        stop_active=stop_active,
                        stop_cycle=stop_cycle,
                    )
                    for start, stop_active, stop_cycle in init_timeslots(
                        times_dt, dur_tdelta_cycle, dur_tdelta_active, n_timeslots, start_dt
                    )
                )

                round_start_time = timeslots[0].start
                round_stop_time = timeslots[-1].stop_cycle
                times_dt = tuple(ts.start for ts in timeslots)

                slots_total = len(timeslots) * len(locations)
                slots_required = n_teams * rnd.rounds_per_team
                slots_empty = slots_total - slots_required

                unfilled_allowed = slots_empty > 0

                yield TournamentRound(
                    roundtype=rnd.roundtype,
                    roundtype_idx=roundtype_idx,
                    rounds_per_team=rnd.rounds_per_team,
                    teams_per_round=rnd.teams_per_round,
                    times=times_dt,
                    start_time=round_start_time,
                    stop_time=round_stop_time,
                    duration_minutes=dur_tdelta_cycle,
                    location_type=rnd.location,
                    locations=locations,
                    num_timeslots=n_timeslots,
                    timeslots=timeslots,
                    slots_total=slots_total,
                    slots_required=slots_required,
                    slots_empty=slots_empty,
                    unfilled_allowed=unfilled_allowed,
                )

        rounds = sorted(_generate_rounds(), key=lambda r: r.start_time)
        return tuple(rounds)

    @classmethod
    def parse_location_config(cls, location_models: tuple[LocationModel, ...]) -> tuple[Location, ...]:
        """Parse and return a list of Location objects from the configuration."""
        location_idx_iter = itertools.count()

        def _generate_locations() -> Iterator[Location]:
            for loctype in location_models:
                for name in range(1, loctype.count + 1):
                    for side_iter in range(1, loctype.sides + 1):
                        yield Location(
                            idx=next(location_idx_iter),
                            locationtype=loctype.name,
                            name=name,
                            side=-1 if loctype.sides == 1 else side_iter,
                            teams_per_round=loctype.sides,
                        )

        return tuple(_generate_locations())

    def log_creation_info(self) -> None:
        """Log information about the application configuration creation."""
        logger.debug("AppConfig created successfully.\n%s", self)
        for r in self.tournament.rounds:
            logger.debug("Initialized tournament round: %s", r)
        logger.debug("Initialized tournament configuration: %s", self.tournament)
        logger.debug("Initialized operator configuration: %s", self.genetic.operator)
        logger.debug("Initialized genetic algorithm parameters: %s", self.genetic.parameters)
        logger.debug("Initialized random number generator: %s.", self.rng.bit_generator)
