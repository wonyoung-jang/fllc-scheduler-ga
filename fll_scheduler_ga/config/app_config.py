"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from ..data_model.event import EventFactory
from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.team import TeamFactory
from ..data_model.time import TimeSlot
from ..genetic.fitness import FitnessEvaluator
from ..operators.crossover import build_crossovers
from ..operators.mutation import build_mutations
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .benchmark import FitnessBenchmark
from .config import Round, RoundType, TournamentConfig
from .constants import RANDOM_SEED_RANGE, CrossoverOp, MutationOp
from .ga_context import GaContext
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
    rng: Random

    @classmethod
    def create_app_config(cls, args: Namespace, path: Path | None = None) -> AppConfig:
        """Create and return the application configuration."""
        if path is None:
            path = Path(args.config_file).resolve()

        parser = AppConfigParser.get_config_parser(args, path)

        return cls(
            tournament=parser.load_tournament_config(),
            operators=parser.load_operator_config(),
            ga_params=parser.load_ga_parameters(),
            rng=parser.load_rng(),
        )

    def create_ga_context(self) -> GaContext:
        """Create and return a GaContext with the provided configuration."""
        rng = self.rng
        tconfig = self.tournament
        team_factory = TeamFactory(tconfig)
        event_factory = EventFactory(tconfig)

        repairer = Repairer(rng, tconfig, event_factory)
        benchmark = FitnessBenchmark(tconfig, event_factory)
        evaluator = FitnessEvaluator(tconfig, benchmark)

        num_objectives = len(evaluator.objectives)
        params = self.ga_params
        pop_size_ref_points = params.population_size * params.num_islands
        total_pop_size = pop_size_ref_points
        nsga3 = NSGA3(
            rng=rng,
            num_objectives=num_objectives,
            total_size=total_pop_size,
            island_size=params.population_size,
        )
        selection = RandomSelect(rng)
        crossovers = tuple(build_crossovers(self, team_factory, event_factory))
        mutations = tuple(build_mutations(self, event_factory))

        return GaContext(
            app_config=self,
            event_factory=event_factory,
            team_factory=team_factory,
            repairer=repairer,
            evaluator=evaluator,
            nsga3=nsga3,
            selection=selection,
            crossovers=crossovers,
            mutations=mutations,
        )


@dataclass(slots=True)
class AppConfigParser(ConfigParser):
    """Parser for the application configuration."""

    args: Namespace
    kwargs: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        super(AppConfigParser, self).__init__(**self.kwargs)

    @classmethod
    def get_config_parser(cls, args: Namespace, path: Path) -> AppConfigParser:
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

    def load_tournament_config(self) -> TournamentConfig:
        """Load and return the tournament configuration from the provided ConfigParser."""
        num_teams, team_ids = self.parse_teams_config()
        Schedule.set_team_identities(team_ids)

        time_fmt = self.parse_time_config()
        TimeSlot.set_time_format(time_fmt)

        rounds, round_requirements = self.parse_rounds_config(num_teams, time_fmt)

        all_rounds_per_team = [r.rounds_per_team for r in rounds]
        total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
        unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1

        weights = self.parse_fitness_config()

        return TournamentConfig(
            num_teams=num_teams,
            time_fmt=time_fmt,
            rounds=rounds,
            round_requirements=round_requirements,
            total_slots=total_slots,
            unique_opponents_possible=unique_opponents_possible,
            weights=weights,
        )

    def parse_teams_config(self) -> tuple[int, dict[int, int | str]]:
        """Parse and return a list of team IDs from the configuration."""
        if not self.has_section("teams"):
            msg = "No 'teams' section found in the configuration file."
            raise KeyError(msg)

        sec = self["teams"]
        num_teams = sec.getint("num_teams", fallback=0)
        identities_raw = sec.get("identities", fallback="").strip()
        identities = [i.strip() for i in identities_raw.split(",") if i.strip()] if identities_raw else []

        if num_teams and not identities:
            identities = [str(i) for i in range(1, num_teams + 1)]
        if identities and not num_teams:
            num_teams = len(identities)

        if num_teams and identities and num_teams != len(identities):
            msg = "Number of teams does not match number of identities."
            raise ValueError(msg)

        team_ids = {
            i: (int(team_id) if team_id.isdigit() else team_id) for i, team_id in enumerate(identities, start=1)
        }
        return num_teams, team_ids

    def parse_time_config(self) -> str:
        """Parse and return the time format configuration."""
        if not self.has_section("time"):
            msg = "No 'time' section found in the configuration file."
            raise KeyError(msg)

        fmt_val = self["time"].getint("format", fallback=None)
        fmt_map = {12: "%I:%M %p", 24: "%H:%M"}
        if fmt_val not in fmt_map:
            msg = "Invalid time format. Must be 12 or 24."
            raise ValueError(msg)
        return fmt_map[fmt_val]

    def parse_rounds_config(self, num_teams: int, time_fmt: str) -> tuple[list[Round], dict[RoundType, int]]:
        """Parse and return a list of Round objects from the configuration.

        Args:
            num_teams (int): The total number of teams in the tournament.
            time_fmt (str): The time format string.

        Returns:
            list[Round]: A list of Round objects parsed from the configuration.
            dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.

        """
        rounds: list[Round] = []
        roundreqs: dict[RoundType, int] = {}
        all_locations = self.load_location_config()
        round_sections = (s for s in self.sections() if s.startswith("round"))

        for section in round_sections:
            self.validate_round_section(section)
            sec = self[section]

            roundtype = sec.get("round_type")
            rounds_per_team = sec.getint("rounds_per_team")
            teams_per_round = sec.getint("teams_per_round")
            duration_minutes = timedelta(minutes=sec.getint("duration_minutes"))

            if start_time := sec.get("start_time"):
                start_time = datetime.strptime(start_time.strip(), time_fmt).replace(tzinfo=UTC)

            if stop_time := sec.get("stop_time"):
                stop_time = datetime.strptime(stop_time.strip(), time_fmt).replace(tzinfo=UTC)

            if times := sec.get("times", fallback=[]):
                times = [datetime.strptime(t.strip(), time_fmt).replace(tzinfo=UTC) for t in times.split(",")]
                start_time = times[0]

            location = sec.get("location")
            locations = sorted(
                (loc for loc in all_locations.values() if loc.name == location),
                key=lambda loc: (loc.identity, loc.side),
            )

            roundreqs.setdefault(roundtype, rounds_per_team)

            rounds.append(
                Round(
                    roundtype=roundtype,
                    rounds_per_team=rounds_per_team,
                    teams_per_round=teams_per_round,
                    times=times,
                    start_time=start_time,
                    stop_time=stop_time,
                    duration_minutes=duration_minutes,
                    num_teams=num_teams,
                    location=location,
                    locations=locations,
                )
            )

        if not rounds:
            msg = "No rounds defined in the configuration file."
            raise ValueError(msg)

        return rounds, roundreqs

    def load_location_config(self) -> dict[tuple[str, int | str, int, int], Location]:
        """Parse and return a dictionary of location IDs to names from the configuration."""
        locations = {}
        for s in (s for s in self.sections() if s.startswith("location")):
            sec = self[s]
            name = sec.get("name")
            sides = sec.getint("sides")
            teams_per_round = sec.getint("teams_per_round")
            identities = sec.get("identities", "")
            for i in (i.strip() for i in identities.split(",") if i.strip()):
                identity = int(i) if i.isdigit() else i
                for j in range(1, sides + 1):
                    side = 0 if sides == 1 else j
                    key = (name, identity, teams_per_round, side)
                    locations[key] = Location(
                        name=name,
                        identity=identity,
                        teams_per_round=teams_per_round,
                        side=side,
                    )
        return locations

    def validate_round_section(self, section: str) -> None:
        """Validate round sections in config file."""
        if not self.has_section(section):
            msg = f"Missing section: '{section}'"
            raise KeyError(msg)

        sec = self[section]
        required = (
            "round_type",
            "rounds_per_team",
            "teams_per_round",
            "duration_minutes",
            "location",
        )
        for option in required:
            if not sec.get(option):
                msg = f"No '{option}' option found in section '{section}'."
                raise KeyError(msg)

        if not sec.get("start_time") and not sec.get("times"):
            msg = f"Either 'start_time' or 'times' must be specified in section '{section}'."
            raise KeyError(msg)

    def parse_fitness_config(self) -> tuple[float, float]:
        """Parse and return fitness-related configuration values."""
        if not self.has_section("fitness"):
            msg = "No 'fitness' section found in the configuration file."
            raise KeyError(msg)

        sec = self["fitness"]
        weights = (
            max(0, sec.getfloat("weight_mean", fallback=3)),
            max(0, sec.getfloat("weight_variation", fallback=1)),
            max(0, sec.getfloat("weight_range", fallback=1)),
        )
        total = sum(weights)
        if total == 0:
            logger.warning("All fitness weights are zero. Using equal weights.")
            return (0.5, 0.5, 0.5)
        return tuple(w / total for w in weights)

    def load_operator_config(self) -> OperatorConfig:
        """Parse and return the operator configuration from the provided ConfigParser."""
        options = {
            ("crossover", "crossover_types", "", ""): (c.value for c in CrossoverOp),
            ("crossover", "crossover_ks", "", "int"): (1, 2, 4, 8),
            ("mutation", "mutation_types", "", ""): (m.value for m in MutationOp),
        }
        operator_config = {}

        for (section, opt, fallback, dtype), default in options.items():
            sec = f"genetic.operator.{section}"
            if self.has_option(sec, opt):
                operator_config[opt] = tuple(self.parse_operator(sec, opt, fallback, dtype))
            else:
                operator_config[opt] = tuple(default)
                logger.warning("%s not found in config. Using defaults: %s", opt, default)

        return OperatorConfig(**operator_config)

    def parse_operator(self, section: str, option: str, fallback: str, dtype: str = "") -> Iterator[str]:
        """Parse a list of operator types from the configuration."""
        raw = self.get(section, option, fallback=fallback)
        items = (i.strip() for i in raw.split(",") if i.strip())
        if dtype == "int":
            yield from (int(i) for i in items)
        yield from items

    def load_ga_parameters(self) -> GaParameters:
        """Build a GaParameters, overriding defaults with any provided CLI args."""
        if not self.has_section("genetic"):
            msg = "No 'genetic' section found in the configuration file."
            raise KeyError(msg)

        sec = self["genetic"]

        params = {
            "population_size": sec.getint("population_size", fallback=16),
            "generations": sec.getint("generations", fallback=128),
            "offspring_size": sec.getint("offspring_size", fallback=12),
            "crossover_chance": sec.getfloat("crossover_chance", fallback=0.5),
            "mutation_chance": sec.getfloat("mutation_chance", fallback=0.5),
            "num_islands": sec.getint("num_islands", fallback=10),
            "migration_interval": sec.getint("migration_interval", fallback=10),
            "migration_size": sec.getint("migration_size", fallback=4),
        }

        for key in params:
            if (cli_val := getattr(self.args, key, None)) is not None:
                params[key] = cli_val

        return GaParameters(**params)

    def load_rng(self) -> Random:
        """Set up the random number generator."""
        seed_val = ""
        if self.args.rng_seed is not None:
            seed_val = self.args.rng_seed
        elif self.has_section("genetic") and self["genetic"].get("seed"):
            seed_val = self["genetic"]["seed"].strip()

        if not seed_val:
            seed_val = Random().randint(*RANDOM_SEED_RANGE)

        try:
            seed = int(seed_val)
        except (TypeError, ValueError):
            seed = abs(hash(seed_val)) % (RANDOM_SEED_RANGE[1] + 1)

        logger.debug("Using RNG seed: %d", seed)
        return Random(seed)
