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
from .constants import CONFIG_FILE, RANDOM_SEED_RANGE
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
    TeamsModel,
    TimeModel,
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
        identities = cls.parse_teams_config(model.teams)
        num_teams = len(identities)
        team_ids = dict(enumerate(identities, start=1))

        time_fmt = cls.parse_time_config(model.time)
        TimeSlot.time_fmt = time_fmt

        if not (locations := cls.parse_location_config(model.locations)):
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)

        if not (rounds := cls.parse_rounds_config(model.rounds, num_teams, time_fmt, locations)):
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        rounds.sort(key=lambda r: (r.start_time))
        roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
        round_str_to_idx = {r.roundtype: r.roundtype_idx for r in rounds}
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}

        total_slots_possible = sum(r.slots_total for r in rounds)
        total_slots_required = sum(r.slots_required for r in rounds)
        unique_opponents_possible = 1 <= max(r.rounds_per_team for r in rounds) <= num_teams - 1

        roundreqs_array = np.tile(list(roundreqs.values()), (num_teams, 1))

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
            num_teams=num_teams,
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
    def _parse_time(cls, raw: str, fmt: str) -> datetime:
        return datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)

    @classmethod
    def parse_teams_config(cls, t_model: TeamsModel) -> list[int | str]:
        """Parse and return a list of team IDs from the configuration."""
        num_teams = t_model.num_teams or 0
        identities = t_model.identities or []

        if num_teams and not identities:
            return [str(i) for i in range(1, num_teams + 1)]

        if not num_teams and identities:
            num_teams = len(identities)

        if num_teams != len(identities):
            msg = f"Number of teams ({num_teams}) does not match number of identities ({len(identities)})."
            raise ValueError(msg)

        return identities

    @classmethod
    def parse_time_config(cls, time_cfg: TimeModel) -> str:
        """Parse and return the time format configuration."""
        fmt_map = {12: "%I:%M %p", 24: "%H:%M"}
        return fmt_map[time_cfg.format]

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
        for rti, sec in enumerate(round_models):
            duration_minutes = timedelta(minutes=sec.duration_minutes)
            start_time_dt = cls._parse_time(sec.start_time, time_fmt)
            times_dt = [cls._parse_time(t, time_fmt) for t in sec.times] if sec.times else []

            locations_in_sec = [loc for loc in locations if loc.locationtype == sec.location]
            locations_in_sec.sort(key=lambda loc: loc.idx)

            num_timeslots = cls.calc_num_timeslots(times_dt, locations_in_sec, num_teams, sec.rounds_per_team)
            timeslots: list[TimeSlot] = []
            for start, stop in cls.init_timeslots(times_dt, duration_minutes, num_timeslots, start_time_dt):
                timeslots.append(TimeSlot(idx=next(timeslot_idx_iter), start=start, stop=stop))

            final_start_time = times_dt[0] if times_dt else timeslots[0].start
            final_stop_time = (times_dt[-1] + duration_minutes) if times_dt else timeslots[-1].stop
            if not times_dt:
                times_dt = [ts.start for ts in timeslots]

            slots_total = len(timeslots) * len(locations_in_sec)
            slots_required = num_teams * sec.rounds_per_team

            tournament_round = TournamentRound(
                roundtype=sec.roundtype,
                roundtype_idx=rti,
                rounds_per_team=sec.rounds_per_team,
                teams_per_round=sec.teams_per_round,
                times=times_dt,
                start_time=final_start_time,
                stop_time=final_stop_time,
                duration_minutes=duration_minutes,
                location_type=sec.location,
                locations=locations_in_sec,
                num_timeslots=num_timeslots,
                timeslots=timeslots,
                slots_total=slots_total,
                slots_required=slots_required,
            )
            tournament_rounds.append(tournament_round)
        return tournament_rounds

    @classmethod
    def calc_num_timeslots(
        cls, times: list[datetime], locations: list[Location], num_teams: int, rounds_per_team: int
    ) -> int:
        """Calculate the number of timeslots needed for a round."""
        if times:
            return len(times)
        if not locations:
            return 0
        return ceil((num_teams * rounds_per_team) / len(locations))

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
            max(0, fitness_model.weight_mean),
            max(0, fitness_model.weight_variation),
            max(0, fitness_model.weight_range),
        )
        total = sum(weights)
        if total == 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(w / total for w in weights)

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
        seed_val = (
            model.genetic.parameters.rng_seed
            if model.genetic.parameters.rng_seed is not None
            else np.random.default_rng().integers(*RANDOM_SEED_RANGE)
        )
        try:
            seed = int(seed_val)
        except (TypeError, ValueError):
            seed = abs(hash(seed_val)) % (RANDOM_SEED_RANGE[1] + 1)

        return np.random.default_rng(seed)

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
