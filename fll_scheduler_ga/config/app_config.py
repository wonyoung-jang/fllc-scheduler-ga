"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

import itertools
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data_model.config import Round, RoundType, TournamentConfig
from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.time import TimeSlot
from .constants import OPERATOR_CONFIG_OPTIONS, RANDOM_SEED_RANGE
from .ga_operators_config import OperatorConfig
from .ga_parameters import GaParameters

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator

logger = getLogger(__name__)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    tournament: TournamentConfig
    operators: OperatorConfig
    ga_params: GaParameters
    rng: np.random.Generator

    @classmethod
    def build(cls, args: Namespace, path: Path | None = None) -> AppConfig:
        """Create and return the application configuration."""
        if path is None:
            path = Path(args.config_file).resolve()

        parser = AppConfigParser.build(args, path)

        tournament_params = parser.load_tournament_config()
        operators_params = parser.load_operator_config()
        ga_params = parser.load_ga_parameters()

        return cls(
            tournament=TournamentConfig.build(tournament_params),
            operators=OperatorConfig.build(operators_params),
            ga_params=GaParameters.build(ga_params),
            rng=parser.load_rng(),
        )


@dataclass(slots=True)
class AppConfigParser(ConfigParser):
    """Parser for the application configuration."""

    args: Namespace
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        super(AppConfigParser, self).__init__(**self.kwargs)

    @classmethod
    def build(cls, args: Namespace, path: Path) -> AppConfigParser:
        """Get a ConfigParser instance for the given config file path."""
        if not Path(path).exists():
            msg = f"Configuration file does not exist at: {path}"
            raise FileNotFoundError(msg)

        config_parser_options = {
            "inline_comment_prefixes": ("#", ";"),
        }
        parser = cls(args, config_parser_options)
        parser.read(path)
        logger.debug("Configuration file loaded from %s", path)
        return parser

    def load_tournament_config(self) -> dict[str, Any]:
        """Load and return the tournament configuration from the provided ConfigParser."""
        num_teams, team_ids = self.parse_teams_config()

        time_fmt = self.parse_time_config()
        TimeSlot.set_time_format(time_fmt)

        locations = self.parse_location_config()
        rounds, roundreqs, round_to_int, round_to_tpr = self.parse_rounds_config(num_teams, time_fmt, locations)
        round_idx_to_tpr = {r.roundtype_idx: r.teams_per_round for r in rounds}

        total_slots_possible = sum(len(r.timeslots) * len(r.locations) for r in rounds)
        total_slots_required = sum(num_teams * r.rounds_per_team for r in rounds)
        unique_opponents_possible = 1 <= max(r.rounds_per_team for r in rounds) <= num_teams - 1

        roundreqs_array_flat = np.array([roundreqs[r.roundtype] for r in rounds], dtype=int)
        roundreqs_array = np.tile(roundreqs_array_flat, (num_teams, 1))

        Schedule.set_team_identities(team_ids)
        Schedule.set_total_num_events(total_slots_possible)
        Schedule.set_team_roundreqs(roundreqs_array)

        weights = self.parse_fitness_config()

        return {
            "num_teams": num_teams,
            "time_fmt": time_fmt,
            "rounds": rounds,
            "round_requirements": roundreqs,
            "round_to_int": round_to_int,
            "round_to_tpr": round_to_tpr,
            "round_idx_to_tpr": round_idx_to_tpr,
            "total_slots_possible": total_slots_possible,
            "total_slots_required": total_slots_required,
            "unique_opponents_possible": unique_opponents_possible,
            "weights": weights,
            "locations": locations,
        }

    def _get_section(self, section: str) -> SectionProxy:
        """Get a section from the config as a dictionary."""
        if not self.has_section(section):
            msg = f"No '{section}' section found in the configuration file."
            raise KeyError(msg)
        return self[section]

    def _iter_sections_prefix(self, prefix: str) -> Iterator[SectionProxy]:
        """Iterate over sections with a specific prefix."""
        yield from (self._get_section(s) for s in self.sections() if s.startswith(prefix))

    def _parse_time(self, raw: str, fmt: str) -> datetime:
        return datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)

    def parse_teams_config(self) -> tuple[int, dict[int, int | str]]:
        """Parse and return a list of team IDs from the configuration."""
        sec = self._get_section("teams")
        num_teams = sec.getint("num_teams", fallback=0)
        identities_raw = sec.get("identities", fallback="").strip()
        identities = [i.strip() for i in identities_raw.split(",") if i.strip()] if identities_raw else []

        if num_teams and not identities:
            identities = [str(i) for i in range(1, num_teams + 1)]

        if not num_teams and identities:
            num_teams = len(identities)

        if num_teams != len(identities):
            msg = f"Number of teams ({num_teams}) does not match number of identities ({len(identities)})."
            raise ValueError(msg)

        team_ids = {i: (int(tid) if tid.isdigit() else tid) for i, tid in enumerate(identities, start=1)}
        return num_teams, team_ids

    def parse_time_config(self) -> str:
        """Parse and return the time format configuration."""
        sec = self._get_section("time")
        fmt_val = sec.getint("format", fallback=None)
        fmt_map = {12: "%I:%M %p", 24: "%H:%M"}
        if fmt_val not in fmt_map:
            msg = "Invalid time format. Must be 12 or 24."
            raise ValueError(msg)
        return fmt_map[fmt_val]

    def parse_rounds_config(
        self, num_teams: int, time_fmt: str, locations: set[Location]
    ) -> tuple[list[Round], dict[RoundType, int], dict[RoundType, int], dict[RoundType, int]]:
        """Parse and return a list of Round objects from the configuration.

        Args:
            num_teams (int): The total number of teams in the tournament.
            time_fmt (str): The time format string.
            locations (set[Location]): A set of Location objects.

        Returns:
            list[Round]: A list of Round objects parsed from the configuration.
            dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.
            dict[RoundType, int]: A mapping of round types to their index in the rounds list.
            dict[RoundType, int]: A mapping of round types to their teams per round.

        """
        rounds: list[Round] = []
        roundreqs: dict[RoundType, int] = {}
        timeslot_idx_iter = itertools.count()

        for rti, sec in enumerate(self._iter_sections_prefix("round")):
            self.validate_round_section(sec)
            roundtype = sec.get("round_type")
            rounds_per_team = sec.getint("rounds_per_team")
            teams_per_round = sec.getint("teams_per_round")
            duration_minutes = timedelta(minutes=sec.getint("duration_minutes"))

            if start_time := sec.get("start_time"):
                start_time = self._parse_time(start_time, time_fmt)

            if stop_time := sec.get("stop_time"):
                stop_time = self._parse_time(stop_time, time_fmt)

            if times := sec.get("times", fallback=[]):
                times = [self._parse_time(t, time_fmt) for t in times.split(",")]
                start_time = times[0]

            location = sec.get("location")
            locations_in_sec = sorted(
                (loc for loc in locations if loc.locationtype == location),
                key=lambda loc: (loc.locationtype, loc.name, loc.side),
            )

            num_timeslots = self.calc_num_timeslots(times, locations_in_sec, num_teams, rounds_per_team)
            timeslots = []
            for start, stop in self.init_timeslots(times, duration_minutes, num_timeslots, start_time):
                timeslots.append(
                    TimeSlot(
                        idx=next(timeslot_idx_iter),
                        start=start,
                        stop=stop,
                    )
                )
            start_time = self.init_start_time(times, timeslots)
            stop_time = self.init_stop_time(times, timeslots, duration_minutes)

            roundreqs[roundtype] = rounds_per_team

            rounds.append(
                Round(
                    roundtype=roundtype,
                    roundtype_idx=rti,
                    rounds_per_team=rounds_per_team,
                    teams_per_round=teams_per_round,
                    times=times,
                    start_time=start_time,
                    stop_time=stop_time,
                    duration_minutes=duration_minutes,
                    num_teams=num_teams,
                    location=location,
                    locations=locations_in_sec,
                    num_timeslots=num_timeslots,
                    timeslots=timeslots,
                )
            )

        if not rounds:
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        rounds.sort(key=lambda r: (r.start_time))
        round_to_int = {r.roundtype: i for i, r in enumerate(rounds)}
        round_to_tpr = {r.roundtype: r.teams_per_round for r in rounds}
        return rounds, roundreqs, round_to_int, round_to_tpr

    def calc_num_timeslots(
        self, times: list[datetime], locations: list[Location], num_teams: int, rounds_per_team: int
    ) -> int:
        """Calculate the number of timeslots needed for a round."""
        if times:
            num_timeslots = len(times)
        elif not locations:
            num_timeslots = 0
        else:
            num_timeslots = ceil((num_teams * rounds_per_team) / len(locations))
        return num_timeslots

    def init_timeslots(
        self, times: list[datetime], dur: timedelta, numslots: int, start: datetime
    ) -> Iterator[tuple[datetime, datetime]]:
        """Initialize the timeslots for the round."""
        start = times[0] if times else start
        for i in range(1, numslots + 1):
            if not times:
                stop = start + dur
            elif i < len(times):
                stop = times[i]
            elif i == len(times):
                stop += dur

            yield (start, stop)
            start = stop

    def init_start_time(self, times: list[datetime], timeslots: list[TimeSlot]) -> datetime:
        """Initialize the start time if not provided."""
        if times:
            return times[0]
        return timeslots[0].start

    def init_stop_time(self, times: list[datetime], timeslots: list[TimeSlot], duration: timedelta) -> datetime:
        """Initialize the stop time if not provided."""
        if times:
            return times[-1] + duration
        return timeslots[-1].stop

    def parse_location_config(self) -> list[Location]:
        """Parse and return a list of Location objects from the configuration."""
        locations: list[Location] = []
        location_idx_iter = itertools.count()

        for sec in self._iter_sections_prefix("location"):
            self.validate_location_section(sec)
            name = sec.get("name")

            sides = sec.getint("sides")
            teams_per_round = sides

            count = sec.getint("count")

            for identifier in range(1, count + 1):
                for j in range(1, sides + 1):
                    side = -1 if sides == 1 else j
                    locations.append(
                        Location(
                            idx=next(location_idx_iter),
                            locationtype=name,
                            name=identifier,
                            side=side,
                            teams_per_round=teams_per_round,
                        )
                    )

        locations.sort(key=lambda loc: (loc.locationtype, loc.name, loc.side))
        locations_log_str = "\n".join(repr(loc) for loc in locations)
        logger.debug("Loaded locations:\n%s", locations_log_str)
        return locations

    def validate_location_section(self, section: SectionProxy) -> None:
        """Validate location sections in config file."""
        required = (
            "name",
            "sides",
            "count",
        )
        for option in required:
            if not section.get(option):
                msg = f"No '{option}' option found in section '{section.name}'."
                raise KeyError(msg)

    def validate_round_section(self, section: SectionProxy) -> None:
        """Validate round sections in config file."""
        required = (
            "round_type",
            "rounds_per_team",
            "teams_per_round",
            "duration_minutes",
            "location",
        )
        for option in required:
            if not section.get(option):
                msg = f"No '{option}' option found in section '{section.name}'."
                raise KeyError(msg)

        if not section.get("start_time") and not section.get("times"):
            msg = f"Either 'start_time' or 'times' must be specified in section '{section.name}'."
            raise KeyError(msg)

    def parse_fitness_config(self) -> tuple[float, float]:
        """Parse and return fitness-related configuration values."""
        sec = self._get_section("fitness")
        self.validate_fitness_section(sec)
        weights = (
            max(0, sec.getfloat("weight_mean", fallback=1)),
            max(0, sec.getfloat("weight_variation", fallback=1)),
            max(0, sec.getfloat("weight_range", fallback=1)),
        )
        total = sum(weights)
        if total == 0:
            logger.warning("All fitness weights are zero. Using equal weights.")
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(w / total for w in weights)

    def validate_fitness_section(self, section: SectionProxy) -> None:
        """Validate fitness sections in config file."""
        required = (
            "weight_mean",
            "weight_variation",
            "weight_range",
        )
        for option in required:
            if not section.get(option):
                msg = f"No '{option}' option found in section '{section.name}'."
                raise KeyError(msg)

    def load_operator_config(self) -> dict[str, Any]:
        """Parse and return the operator configuration from the provided ConfigParser."""
        params = {}
        for (section, opt, fallback, dtype), default in OPERATOR_CONFIG_OPTIONS.items():
            sec = f"genetic.operator.{section}"
            if self.has_option(sec, opt):
                params[opt] = self.parse_operator(sec, opt, fallback, dtype)
            else:
                params[opt] = default
                logger.warning("%s not found in config. Using defaults: %s", opt, default)
        return params

    def parse_operator(self, section: str, option: str, fallback: str, dtype: str = "") -> tuple[str]:
        """Parse a list of operator types from the configuration."""
        raw = self.get(section, option, fallback=fallback)
        items = (i.strip() for i in raw.split(",") if i.strip())
        if dtype == "int":
            return tuple(int(i) for i in items)
        return tuple(items)

    def load_ga_parameters(self) -> dict[str, Any]:
        """Build a GaParameters, overriding defaults with any provided CLI args."""
        sec = self._get_section("genetic")
        params = {
            "population_size": sec.getint("population_size"),
            "generations": sec.getint("generations"),
            "offspring_size": sec.getint("offspring_size"),
            "crossover_chance": sec.getfloat("crossover_chance"),
            "mutation_chance": sec.getfloat("mutation_chance"),
            "num_islands": sec.getint("num_islands"),
            "migration_interval": sec.getint("migration_interval"),
            "migration_size": sec.getint("migration_size"),
        }
        for k in params:
            if v := getattr(self.args, k, None):
                params[k] = v
        return params

    def load_rng(self) -> np.random.Generator:
        """Set up the random number generator."""
        sec = self._get_section("genetic")
        seed_val = ""
        rng_seed = getattr(self.args, "rng_seed", None)
        if rng_seed is not None:
            seed_val = rng_seed
        elif sec.get("seed"):
            seed_val = sec["seed"].strip()
        else:
            seed_val = np.random.default_rng().integers(*RANDOM_SEED_RANGE)

        try:
            seed = int(seed_val)
        except (TypeError, ValueError):
            seed = abs(hash(seed_val)) % (RANDOM_SEED_RANGE[1] + 1)

        logger.debug("Using RNG seed: %d", seed)
        return np.random.default_rng(seed)
