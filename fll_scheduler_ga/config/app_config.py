"""Configuration for the FLL Scheduler GA application."""

from __future__ import annotations

from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, field
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
        nsga3 = NSGA3(
            rng=rng,
            num_objectives=num_objectives,
            total_size=pop_size_ref_points,
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
    kwargs: dict[str, Any] = field(default_factory=dict)

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
        idents_raw = sec.get("identities", fallback="").strip()
        idents = [i.strip() for i in idents_raw.split(",") if i.strip()] if idents_raw else []

        if num_teams and not idents:
            idents = [str(i) for i in range(1, num_teams + 1)]

        if not num_teams and idents:
            num_teams = len(idents)

        if num_teams != len(idents):
            msg = f"Number of teams ({num_teams}) does not match number of identities ({len(idents)})."
            raise ValueError(msg)

        team_ids = {i: (int(tid) if tid.isdigit() else tid) for i, tid in enumerate(idents, start=1)}
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

        for sec in self._iter_sections_prefix("round"):
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
            locations = sorted(
                (loc for loc in all_locations if loc.name == location),
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

    def load_location_config(self) -> set[Location]:
        """Parse and return a set of Location objects from the configuration."""
        locations = set()
        for sec in self._iter_sections_prefix("location"):
            self.validate_location_section(sec)
            name = sec.get("name")
            sides = sec.getint("sides")
            teams_per_round = sec.getint("teams_per_round")
            identities_raw = sec.get("identities", "")
            identities_iter = (i.strip() for i in identities_raw.split(",") if i.strip())
            for ident in identities_iter:
                identity = int(ident) if ident.isdigit() else ident
                for j in range(1, sides + 1):
                    side = 0 if sides == 1 else j
                    locations.add(
                        Location(
                            name=name,
                            identity=identity,
                            teams_per_round=teams_per_round,
                            side=side,
                        )
                    )
        return locations

    def validate_location_section(self, section: SectionProxy) -> None:
        """Validate location sections in config file."""
        required = (
            "name",
            "sides",
            "teams_per_round",
            "identities",
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
        weights = (
            max(0, sec.getfloat("weight_mean", fallback=3)),
            max(0, sec.getfloat("weight_variation", fallback=1)),
            max(0, sec.getfloat("weight_range", fallback=1)),
        )
        total = sum(weights)
        if total == 0 or len(set(weights)) == 1:
            logger.warning("All fitness weights are zero or identical. Using equal weights.")
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(w / total for w in weights)

    def load_operator_config(self) -> OperatorConfig:
        """Parse and return the operator configuration from the provided ConfigParser."""
        options = {
            ("crossover", "crossover_types", "", ""): (c.value for c in CrossoverOp),
            ("crossover", "crossover_ks", "", "int"): (1, 2, 4, 8),
            ("mutation", "mutation_types", "", ""): (m.value for m in MutationOp),
        }
        params = {}
        for (section, opt, fallback, dtype), default in options.items():
            sec = f"genetic.operator.{section}"
            if self.has_option(sec, opt):
                params[opt] = tuple(self.parse_operator(sec, opt, fallback, dtype))
            else:
                params[opt] = tuple(default)
                logger.warning("%s not found in config. Using defaults: %s", opt, default)

        return OperatorConfig(**params)

    def parse_operator(self, section: str, option: str, fallback: str, dtype: str = "") -> Iterator[str]:
        """Parse a list of operator types from the configuration."""
        raw = self.get(section, option, fallback=fallback)
        items = (i.strip() for i in raw.split(",") if i.strip())
        if dtype == "int":
            yield from (int(i) for i in items)
        yield from items

    def load_ga_parameters(self) -> GaParameters:
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

        return GaParameters(**params)

    def load_rng(self) -> Random:
        """Set up the random number generator."""
        sec = self._get_section("genetic")
        seed_val = ""
        if self.args.rng_seed is not None:
            seed_val = self.args.rng_seed
        elif sec.get("seed"):
            seed_val = sec["seed"].strip()
        else:
            seed_val = Random().randint(*RANDOM_SEED_RANGE)

        try:
            seed = int(seed_val)
        except (TypeError, ValueError):
            seed = abs(hash(seed_val)) % (RANDOM_SEED_RANGE[1] + 1)

        logger.debug("Using RNG seed: %d", seed)
        return Random(seed)
