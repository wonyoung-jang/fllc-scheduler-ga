"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

import itertools
import json
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.timeslot import TimeSlot
from .constants import CONFIG_FILE, TIME_FORMAT_MAP
from .schemas import (
    AppConfigModel,
    ExportModel,
    GeneticModel,
    ImportModel,
    LocationModel,
    LoggingModel,
    RoundModel,
    RuntimeModel,
    TournamentConfig,
    TournamentRound,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    """Configuration for the FLL Scheduler GA application."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    runtime: RuntimeModel
    imports: ImportModel
    exports: ExportModel
    logging: LoggingModel
    genetic: GeneticModel
    tournament: TournamentConfig
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
    def build_from_model(cls, model: AppConfigModel) -> AppConfig:
        """Create and return the application configuration from a Pydantic model."""
        model.exports.team_identities = model.teams.get_team_ids()
        return AppConfig(
            runtime=model.runtime,
            imports=model.imports,
            exports=model.exports,
            logging=model.logging,
            genetic=model.genetic,
            tournament=cls.load_tournament_config(model),
            rng=np.random.default_rng(model.genetic.parameters.rng_seed),
        )

    @classmethod
    def load_tournament_config(cls, model: AppConfigModel) -> TournamentConfig:
        """Load and return the tournament configuration from the validated model."""
        teams = model.teams
        time_fmt = ""

        all_time_strs = []
        for rnd in model.rounds:
            if rnd.start_time:
                all_time_strs.append(rnd.start_time)
            if rnd.stop_time:
                all_time_strs.append(rnd.stop_time)
            all_time_strs.extend(rnd.times)

        inferred_formats = [cls.infer_time_format(t) for t in all_time_strs if t]
        time_fmt = ""
        if inferred_formats:
            format_counts = Counter(inferred_formats)
            if len(format_counts) == 1:
                time_fmt = format_counts.most_common(1)[0][0]
            else:
                msg = "Conflicting time formats found in configuration times."
                raise ValueError(msg)

        TimeSlot.time_fmt = time_fmt

        if not (locations := cls.parse_location_config(model.locations)):
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)

        if not (rounds := cls.parse_rounds_config(model.rounds, len(teams), time_fmt, locations)):
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        rounds.sort(key=lambda r: r.start_time)
        roundreqs = {r.roundtype: r.rounds_per_team for r in rounds}
        roundreqs_array = np.tile(list(roundreqs.values()), (len(teams), 1))
        round_str_to_idx = {r.roundtype: r.roundtype_idx for r in rounds}
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}
        total_slots_possible = sum(r.slots_total for r in rounds)
        total_slots_required = sum(r.slots_required for r in rounds)
        unique_opponents_possible = 1 <= max(r.rounds_per_team for r in rounds) <= len(teams) - 1

        Schedule.total_num_events = total_slots_possible
        Schedule.team_roundreqs_array = roundreqs_array

        weights = model.fitness.get_fitness_tuple()

        all_locations = itertools.chain.from_iterable(r.locations for r in rounds)
        all_locations = sorted(all_locations, key=lambda loc: loc.idx)

        all_timeslots = itertools.chain.from_iterable(r.timeslots for r in rounds)
        all_timeslots = sorted(all_timeslots, key=lambda ts: ts.idx)

        max_events_per_team = sum(r.rounds_per_team for r in rounds)
        starts_of_rounds = [r.start_time for r in rounds]
        ends_of_rounds = [r.stop_time for r in rounds]
        round_timeslots = []
        for start, stop in zip(starts_of_rounds, ends_of_rounds, strict=True):
            round_ts = TimeSlot(idx=0, start=start, stop=stop)
            round_timeslots.append(round_ts)

        is_interleaved = any(
            round_timeslots[i].overlaps(round_timeslots[j])
            for i in range(len(round_timeslots))
            for j in range(i + 1, len(round_timeslots))
        )

        return TournamentConfig(
            num_teams=len(teams),
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
            is_interleaved=is_interleaved,
        )

    @classmethod
    def _parse_time(cls, raw: str | None, fmt: str) -> datetime | None:
        if not raw:
            return None
        return datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)

    @classmethod
    def parse_rounds_config(
        cls, round_models: list[RoundModel], num_teams: int, time_fmt: str, locations: list[Location]
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

            dur_raw = rnd.duration_minutes
            dur_valid = cls.validate_duration(start_dt, stop_dt, times_dt, dur_raw, n_timeslots)
            dur_tdelta = timedelta(minutes=dur_valid)

            timeslots = [
                TimeSlot(idx=next(timeslot_idx_iter), start=start, stop=stop)
                for start, stop in cls.init_timeslots(times_dt, dur_tdelta, n_timeslots, start_dt)
            ]

            final_start_time = timeslots[0].start
            final_stop_time = timeslots[-1].stop

            times_dt = times_dt if times_dt else [ts.start for ts in timeslots]

            slots_total = len(timeslots) * len(locations_in_sec)
            slots_required = num_teams * rnd.rounds_per_team
            slots_empty = slots_total - slots_required

            unfilled_allowed = slots_empty > 0

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
                slots_empty=slots_empty,
                unfilled_allowed=unfilled_allowed,
            )
            tournament_rounds.append(tournament_round)
        return tournament_rounds

    @classmethod
    def validate_duration(
        cls, start_dt: datetime | None, stop_dt: datetime | None, times_dt: list[datetime], dur: int, n_timeslots: int
    ) -> int | None:
        """Validate the times configuration for a round.

        Valid conditions:
        1. start + duration
        2. times + duration
        3. start + stop (need to calculate num_timeslots)
        """
        if (start_dt or times_dt) and dur:
            return dur

        if start_dt and stop_dt:
            total_available = (stop_dt - start_dt).total_seconds()
            minimum_duration = total_available // n_timeslots
            return max(1, minimum_duration // 60)

        return None

    @classmethod
    def infer_time_format(cls, time_str: str) -> str | None:
        """Infer the time format from a sample time string."""
        for fmt in TIME_FORMAT_MAP.values():
            try:
                datetime.strptime(time_str.strip(), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
            else:
                return fmt
        return None

    @classmethod
    def calc_num_timeslots(cls, n_times: int, n_locs: int, n_teams: int, rounds_per_team: int) -> int:
        """Calculate the number of timeslots needed for a round."""
        if not n_times and not n_locs:
            msg = "Cannot calculate number of timeslots without times or locations."
            raise ValueError(msg)

        if n_times:
            num_timeslots = n_times
        elif n_locs:
            num_timeslots = ceil((n_teams * rounds_per_team) / n_locs)

        return num_timeslots

    @classmethod
    def init_timeslots(
        cls, start_times: list[datetime], dur_tdelta: timedelta, n_timeslots: int, start_dt: datetime
    ) -> Iterator[tuple[datetime, datetime]]:
        """Initialize the timeslots for the round."""
        if start_times and dur_tdelta:
            stop_times = start_times[1:]
            stop_times.append(stop_times[-1] + dur_tdelta)
            time_pairs = zip(start_times, stop_times, strict=True)
            yield from time_pairs
            return

        current = start_dt
        for _ in range(n_timeslots):
            stop = current + dur_tdelta
            yield (current, stop)
            current = stop

    @classmethod
    def parse_location_config(cls, location_models: list[LocationModel]) -> list[Location]:
        """Parse and return a list of Location objects from the configuration."""
        locations = []
        location_idx_iter = itertools.count()
        for loctype in location_models:
            for name in range(1, loctype.count + 1):
                for side_iter in range(1, loctype.sides + 1):
                    location = Location(
                        idx=next(location_idx_iter),
                        locationtype=loctype.name,
                        name=name,
                        side=-1 if loctype.sides == 1 else side_iter,
                        teams_per_round=loctype.sides,
                    )
                    locations.append(location)
        return locations

    def log_creation_info(self) -> None:
        """Log information about the application configuration creation."""
        logger.debug("AppConfig created successfully.")
        logger.debug("Initialized argument configuration: %s", self.runtime)
        for r in self.tournament.rounds:
            logger.debug("Initialized tournament round: %s", r)
        if sum(self.tournament.weights) == 0:
            logger.debug("All fitness weights are zero; using equal weights.")
        logger.debug("Initialized tournament configuration: %s", self.tournament)
        logger.debug("Initialized operator configuration: %s", self.genetic.operator)
        logger.debug("Initialized genetic algorithm parameters: %s", self.genetic.parameters)
        logger.debug("Initialized random number generator: %s.", self.rng.bit_generator.state["state"])
