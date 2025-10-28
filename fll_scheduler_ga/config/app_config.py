"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.time import TimeSlot
from .constants import CONFIG_FILE, TIME_FORMAT_MAP
from .schemas import (
    AppConfigModel,
    ExportModel,
    FitnessModel,
    GaParameters,
    LocationModel,
    LoggingModel,
    OperatorConfig,
    RoundModel,
    RuntimeModel,
    TournamentConfig,
    TournamentRound,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = getLogger(__name__)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    runtime: RuntimeModel
    exports: ExportModel
    logging: LoggingModel
    tournament: TournamentConfig
    operators: OperatorConfig
    ga_params: GaParameters
    rng: np.random.Generator

    @classmethod
    def build(cls, path: Path | None = None) -> AppConfig:
        """Create and return the application configuration."""
        if path is None:
            path = CONFIG_FILE.resolve()

        if not path.exists():
            msg = f"Configuration file does not exist at: {path}"
            raise FileNotFoundError(msg)

        with path.open(encoding="utf-8") as jf:
            config_data = json.load(jf)

        config_model = AppConfigModel.model_validate(config_data)
        return cls.build_from_model(config_model)

    @classmethod
    def build_from_model(cls, config_model: AppConfigModel) -> AppConfig:
        """Create and return the application configuration from a Pydantic model."""
        tournament_config = cls.load_tournament_config(config_model)
        operator_config = cls.load_operator_config(config_model)
        ga_parameters = cls.load_ga_parameters(config_model)
        rng = cls.load_rng(config_model)
        return cls(
            runtime=config_model.runtime,
            exports=config_model.exports,
            logging=config_model.logging,
            tournament=tournament_config,
            operators=operator_config,
            ga_params=ga_parameters,
            rng=rng,
        )

    @classmethod
    def load_tournament_config(cls, model: AppConfigModel) -> TournamentConfig:
        """Load and return the tournament configuration from the validated model."""
        teams = model.teams.teams
        n_teams = len(teams)
        team_ids = dict(enumerate(teams, start=1))

        time_fmt = TIME_FORMAT_MAP[model.time.format]
        TimeSlot.time_fmt = time_fmt

        if not (locations := cls.parse_location_config(model.locations)):
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)

        if not (rounds := cls.parse_rounds_config(model.rounds, n_teams, time_fmt, locations)):
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        rounds.sort(key=lambda r: (r.start_time))
        roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
        roundreqs_array = np.tile(list(roundreqs.values()), (n_teams, 1))
        round_str_to_idx = {r.roundtype: r.roundtype_idx for r in rounds}
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}
        total_slots_possible = sum(r.slots_total for r in rounds)
        total_slots_required = sum(r.slots_required for r in rounds)
        unique_opponents_possible = 1 <= max(r.rounds_per_team for r in rounds) <= n_teams - 1

        Schedule.team_identities = team_ids
        Schedule.total_num_events = total_slots_possible
        Schedule.team_roundreqs_array = roundreqs_array

        weights = cls.parse_fitness_config(model.fitness)

        all_locations = itertools.chain.from_iterable(r.locations for r in rounds)
        all_locations = sorted(all_locations, key=lambda loc: loc.idx)

        all_timeslots = itertools.chain.from_iterable(r.timeslots for r in rounds)
        all_timeslots = sorted(all_timeslots, key=lambda ts: ts.idx)

        max_events_per_team = sum(r.rounds_per_team for r in rounds)

        return TournamentConfig(
            num_teams=n_teams,
            time_fmt=time_fmt,
            rounds=rounds,
            roundreqs=roundreqs,
            round_str_to_idx=round_str_to_idx,
            round_idx_to_tpr=round_idx_to_tpr,
            total_slots_possible=total_slots_possible,
            total_slots_required=total_slots_required,
            unique_opponents_possible=unique_opponents_possible,
            weights=weights,
            all_locations=all_locations,
            all_timeslots=all_timeslots,
            max_events_per_team=max_events_per_team,
        )

    @classmethod
    def _parse_time(cls, raw: str | None, fmt: str) -> datetime | None:
        if not raw:
            return None
        return datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)

    @classmethod
    def parse_rounds_config(
        cls,
        round_models: list[RoundModel],
        num_teams: int,
        time_fmt: str,
        locations: list[Location],
    ) -> list[TournamentRound]:
        """Parse and return a list of TournamentRound objects from the configuration."""
        tournament_rounds = []
        timeslot_idx_iter = itertools.count()
        for roundtype_idx, rnd in enumerate(round_models):
            start_dt = cls._parse_time(rnd.start_time, time_fmt)
            stop_dt = cls._parse_time(rnd.stop_time, time_fmt)
            times_dt = [cls._parse_time(t, time_fmt) for t in rnd.times] if rnd.times else []

            locations_in_sec = [loc for loc in locations if loc.locationtype == rnd.location]
            locations_in_sec.sort(key=lambda loc: loc.idx)

            n_timeslots = cls.calc_num_timeslots(len(times_dt), len(locations_in_sec), num_teams, rnd.rounds_per_team)

            dur = rnd.duration_minutes
            dur_valid = cls.validate_duration(start_dt, stop_dt, times_dt, dur, n_timeslots)
            dur_tdelta = timedelta(minutes=dur_valid)

            timeslots = [
                TimeSlot(next(timeslot_idx_iter), start, stop)
                for start, stop in cls.init_timeslots(times_dt, dur_tdelta, n_timeslots, start_dt)
            ]

            final_start_time = times_dt[0] if times_dt else timeslots[0].start
            final_stop_time = (times_dt[-1] + dur_tdelta) if times_dt else timeslots[-1].stop

            if not times_dt:
                times_dt = [ts.start for ts in timeslots]

            slots_total = len(timeslots) * len(locations_in_sec)
            slots_required = num_teams * rnd.rounds_per_team

            tournament_round = TournamentRound(
                roundtype=rnd.roundtype,
                roundtype_idx=roundtype_idx,
                rounds_per_team=rnd.rounds_per_team,
                teams_per_round=rnd.teams_per_round,
                times=times_dt,
                start_time=final_start_time,
                stop_time=final_stop_time,
                duration_minutes=dur_tdelta,
                location_type=rnd.location,
                locations=locations_in_sec,
                num_timeslots=n_timeslots,
                timeslots=timeslots,
                slots_total=slots_total,
                slots_required=slots_required,
            )
            tournament_rounds.append(tournament_round)
        return tournament_rounds

    @classmethod
    def validate_duration(
        cls,
        start_time_dt: datetime | None,
        stop_time_dt: datetime | None,
        times_dt: list[datetime],
        dur: int,
        n_timeslots: int,
    ) -> int | None:
        """Validate the times configuration for a round."""
        # Valid conditions:
        #   1. start_time + duration
        #   2. times + duration
        #   3. start_time + stop_time (need to calculate num_timeslots)
        if (start_time_dt or times_dt) and dur:
            return dur

        if start_time_dt and stop_time_dt:
            total_available = (stop_time_dt - start_time_dt).total_seconds()
            minimum_duration = total_available // n_timeslots
            return max(1, minimum_duration // 60)

        return None

    @classmethod
    def calc_num_timeslots(cls, n_times: int, n_locs: int, n_teams: int, rounds_per_team: int) -> int:
        """Calculate the number of timeslots needed for a round."""
        if n_times:
            num_timeslots = n_times
        elif n_locs:
            num_timeslots = ceil((n_teams * rounds_per_team) / n_locs)

        if not n_times and not n_locs:
            msg = "Cannot calculate number of timeslots without times or locations."
            raise ValueError(msg)

        return num_timeslots

    @classmethod
    def init_timeslots(
        cls, times: list[datetime], dur: timedelta, numslots: int, start: datetime
    ) -> Iterator[tuple[datetime, datetime]]:
        """Initialize the timeslots for the round."""
        current_start = times[0] if times else start
        for i in range(1, numslots + 1):
            if not times:
                stop = current_start + dur
            elif i < len(times):
                stop = times[i]
            else:  # Last slot in a list of times needs a duration
                stop = current_start + dur

            yield (current_start, stop)
            current_start = stop

    @classmethod
    def parse_location_config(cls, location_models: list[LocationModel]) -> list[Location]:
        """Parse and return a list of Location objects from the configuration."""
        locations = []
        location_idx_iter = itertools.count()
        for sec in location_models:
            for identifier in range(1, sec.count + 1):
                for j in range(1, sec.sides + 1):
                    side = -1 if sec.sides == 1 else j
                    locations.append(
                        Location(
                            idx=next(location_idx_iter),
                            locationtype=sec.name,
                            name=identifier,
                            side=side,
                            teams_per_round=sec.sides,
                        )
                    )
        return locations

    @classmethod
    def parse_fitness_config(cls, fitness_model: FitnessModel) -> tuple[float, ...]:
        """Parse and return fitness-related configuration values."""
        weights = (
            fitness_model.weight_mean,
            fitness_model.weight_variation,
            fitness_model.weight_range,
        )
        return tuple(w / sum(weights) for w in weights)

    @classmethod
    def load_operator_config(cls, model: AppConfigModel) -> OperatorConfig:
        """Parse and return the operator configuration from the validated model."""
        op_model = model.genetic.operator
        return OperatorConfig(
            crossover_types=op_model.crossover.types,
            crossover_ks=op_model.crossover.k_vals,
            mutation_types=op_model.mutation.types,
        )

    @classmethod
    def load_ga_parameters(cls, model: AppConfigModel) -> GaParameters:
        """Build a GaParameters from the validated model."""
        pa_model = model.genetic.parameters
        return GaParameters(
            population_size=pa_model.population_size,
            generations=pa_model.generations,
            offspring_size=pa_model.offspring_size,
            crossover_chance=pa_model.crossover_chance,
            mutation_chance=pa_model.mutation_chance,
            num_islands=pa_model.num_islands,
            migrate_interval=pa_model.migration_interval,
            migrate_size=pa_model.migration_size,
        )

    @classmethod
    def load_rng(cls, model: AppConfigModel) -> np.random.Generator:
        """Set up the random number generator."""
        return np.random.default_rng(model.genetic.parameters.rng_seed)

    def log_creation_info(self) -> None:
        """Log information about the application configuration creation."""
        logger.debug("AppConfig created successfully.")
        logger.debug("Initialized argument configuration: %s", self.runtime)
        for r in self.tournament.rounds:
            logger.debug("Initialized tournament round: %s", r)
        if sum(self.tournament.weights) == 0:
            logger.debug("All fitness weights are zero; using equal weights.")
        logger.debug("Initialized tournament configuration: %s", self.tournament)
        logger.debug("Initialized operator configuration: %s", self.operators)
        logger.debug("Initialized genetic algorithm parameters: %s", self.ga_params)
        logger.debug("Initialized random number generator: %s.", self.rng.bit_generator.state["state"])
