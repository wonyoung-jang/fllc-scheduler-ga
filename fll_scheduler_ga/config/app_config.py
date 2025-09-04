"""Configuration for the FLL Scheduler GA application."""

from argparse import Namespace
from collections.abc import Iterator
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import getLogger
from pathlib import Path
from random import Random

from ..data_model.location import Location
from ..data_model.schedule import Schedule
from ..data_model.time import TimeSlot
from .config import Round, RoundType, TournamentConfig
from .constants import RANDOM_SEED_RANGE, CrossoverOp, MutationOp, SelectionOp
from .ga_operators_config import OperatorConfig
from .ga_parameters import GaParameters

logger = getLogger(__name__)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    tournament: TournamentConfig
    operators: OperatorConfig
    ga_params: GaParameters
    rng: Random


def create_app_config(args: Namespace, path: Path | None = None) -> AppConfig:
    """Create and return the application configuration."""
    if path is None:
        path = Path(args.config_file).resolve()

    parser = get_config_parser(path)

    return AppConfig(
        tournament=load_tournament_config(parser),
        operators=load_operator_config(parser),
        ga_params=load_ga_parameters(args, parser),
        rng=load_rng(args, parser),
    )


def load_tournament_config(p: ConfigParser) -> TournamentConfig:
    """Load and return the tournament configuration from the provided ConfigParser."""
    num_teams, team_ids = _parse_teams_config(p)
    Schedule.set_team_identities(team_ids)

    time_fmt = _parse_time_config(p)
    TimeSlot.set_time_format(time_fmt)

    rounds, round_requirements = _parse_rounds_config(p, num_teams, time_fmt)

    all_rounds_per_team = [r.rounds_per_team for r in rounds]
    total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
    unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1

    weights = _parse_fitness_config(p)

    return TournamentConfig(
        num_teams=num_teams,
        time_fmt=time_fmt,
        rounds=rounds,
        round_requirements=round_requirements,
        total_slots=total_slots,
        unique_opponents_possible=unique_opponents_possible,
        weights=weights,
    )


def load_operator_config(p: ConfigParser) -> OperatorConfig:
    """Parse and return the operator configuration from the provided ConfigParser.

    Args:
        p (ConfigParser): The ConfigParser instance with operator configuration.

    Returns:
        OperatorConfig: The parsed operator configuration.

    """
    options = {
        ("selection", "selection_types", "", ""): (s.value for s in SelectionOp),
        ("crossover", "crossover_types", "", ""): (c.value for c in CrossoverOp),
        ("crossover", "crossover_ks", "", "int"): (1, 2, 4, 8),
        ("mutation", "mutation_types", "", ""): (m.value for m in MutationOp),
    }
    operator_config = {}
    for (section, opt, fallback, dtype), default in options.items():
        sec = f"genetic.operator.{section}"
        if p.has_option(sec, opt):
            operator_config[opt] = tuple(parse_operator(p, sec, opt, fallback, dtype))
        else:
            operator_config[opt] = tuple(default)
            logger.warning("%s not found in config. Using defaults: %s", opt, default)
    return OperatorConfig(**operator_config)


def load_ga_parameters(args: Namespace, p: ConfigParser) -> GaParameters:
    """Build a GaParameters, overriding defaults with any provided CLI args.

    Args:
        args (Namespace): Parsed command-line arguments.
        p (ConfigParser): Configuration parser with default values.

    Returns:
        GaParameters: Parameters for the genetic algorithm.

    """
    if not p.has_section("genetic"):
        msg = "No 'genetic' section found in the configuration file."
        raise KeyError(msg)

    sec = p["genetic"]

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
        if (cli_val := getattr(args, key, None)) is not None:
            params[key] = cli_val

    return GaParameters(**params)


def load_rng(args: Namespace, p: ConfigParser) -> Random:
    """Set up the random number generator."""
    seed_val = ""
    if args.rng_seed is not None:
        seed_val = args.rng_seed
    elif p.has_section("genetic") and p["genetic"].get("seed"):
        seed_val = p["genetic"]["seed"].strip()

    if not seed_val:
        seed_val = Random().randint(*RANDOM_SEED_RANGE)

    try:
        seed = int(seed_val)
    except (TypeError, ValueError):
        seed = abs(hash(seed_val)) % (RANDOM_SEED_RANGE[1] + 1)

    logger.debug("Using RNG seed: %d", seed)
    return Random(seed)


def get_config_parser(path: Path | None = None) -> ConfigParser:
    """Get a ConfigParser instance for the given config file path.

    Args:
        path (Path | None): Path to the configuration file.

    Returns:
        ConfigParser: The configured ConfigParser instance.

    """
    if path is None:
        path = Path("fll_scheduler_ga/config.ini").resolve()

    if not Path(path).exists():
        msg = f"Configuration file does not exist at: {path}"
        raise FileNotFoundError(msg)

    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    logger.debug("Configuration file loaded from %s", path)
    return parser


def _parse_teams_config(p: ConfigParser) -> tuple[int, dict[int, int | str]]:
    """Parse and return a list of team IDs from the configuration."""
    if not p.has_section("teams"):
        msg = "No 'teams' section found in the configuration file."
        raise KeyError(msg)

    sec = p["teams"]
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

    team_ids = {i: (int(team_id) if team_id.isdigit() else team_id) for i, team_id in enumerate(identities, start=1)}
    return num_teams, team_ids


def _parse_time_config(p: ConfigParser) -> str:
    if not p.has_section("time"):
        msg = "No 'time' section found in the configuration file."
        raise KeyError(msg)

    fmt_val = p["time"].getint("format", fallback=None)
    fmt_map = {12: "%I:%M %p", 24: "%H:%M"}
    if fmt_val not in fmt_map:
        msg = "Invalid time format. Must be 12 or 24."
        raise ValueError(msg)
    return fmt_map[fmt_val]


def _parse_fitness_config(p: ConfigParser) -> tuple[float, float]:
    """Parse and return fitness-related configuration values."""
    if not p.has_section("fitness"):
        msg = "No 'fitness' section found in the configuration file."
        raise KeyError(msg)

    sec = p["fitness"]
    weights = (
        max(0, sec.getfloat("weight_mean", fallback=3)),
        max(0, sec.getfloat("weight_variation", fallback=1)),
    )
    total = sum(weights)
    if total == 0:
        logger.warning("All fitness weights are zero. Using equal weights.")
        return (0.5, 0.5)
    return tuple(w / total for w in weights)


def _parse_rounds_config(
    p: ConfigParser,
    num_teams: int,
    time_fmt: str,
) -> tuple[list[Round], dict[RoundType, int]]:
    """Parse and return a list of Round objects from the configuration.

    Args:
        p (ConfigParser): The ConfigParser instance with tournament configuration.
        num_teams (int): The total number of teams in the tournament.
        time_fmt (str): The time format string.

    Returns:
        list[Round]: A list of Round objects parsed from the configuration.
        dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.

    """
    rounds: list[Round] = []
    roundreqs: dict[RoundType, int] = {}
    all_locations = load_location_config(p)
    round_sections = (s for s in p.sections() if s.startswith("round"))

    for section in round_sections:
        _validate_round_section(p, section)
        sec = p[section]

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


def _validate_round_section(p: ConfigParser, section: str) -> None:
    """Validate round sections in config file."""
    if not p.has_section(section):
        msg = f"Missing section: '{section}'"
        raise KeyError(msg)

    sec = p[section]
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


def parse_operator(
    p: ConfigParser,
    section: str,
    option: str,
    fallback: str,
    dtype: str = "",
) -> Iterator[str]:
    """Parse a list of operator types from the configuration."""
    raw = p.get(section, option, fallback=fallback)
    items = (i.strip() for i in raw.split(",") if i.strip())
    if dtype == "int":
        yield from (int(i) for i in items)
    yield from items


def load_location_config(p: ConfigParser) -> dict[tuple[str, int | str, int, int], Location]:
    """Parse and return a dictionary of location IDs to names from the configuration."""
    locations = {}
    for s in (s for s in p.sections() if s.startswith("location")):
        sec = p[s]
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
