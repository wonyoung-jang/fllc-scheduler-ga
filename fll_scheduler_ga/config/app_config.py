"""Configuration for the FLL Scheduler GA application."""

from argparse import Namespace
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

    parser = _get_config_parser(path)
    tournament = load_tournament_config(parser)
    operators = load_operator_config(parser)
    ga_params = load_ga_parameters(args, parser)
    rng = load_rng(args, parser)
    return AppConfig(tournament=tournament, operators=operators, ga_params=ga_params, rng=rng)


def load_tournament_config(p: ConfigParser) -> TournamentConfig:
    """Load and return the tournament configuration from the provided ConfigParser."""
    num_teams, team_ids = _parse_teams_config(p)
    time_fmt = _parse_time_config(p)
    TimeSlot.set_time_format(time_fmt)
    rounds, round_requirements = _parse_rounds_config(p, num_teams, time_fmt)
    all_rounds_per_team = [r.rounds_per_team for r in rounds]
    total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
    unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1
    weights = _parse_fitness_config(p)
    Schedule.set_team_identities(team_ids)
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
    if p.has_option("genetic.operator.selection", "selection_types"):
        selection_types = _parse_operator_types(p, "genetic.operator.selection", "selection_types", "")
    else:
        selection_types = [
            SelectionOp.TOURNAMENT_SELECT,
            SelectionOp.RANDOM_SELECT,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "selection_types", selection_types)

    if p.has_option("genetic.operator.crossover", "crossover_types"):
        crossover_types = _parse_operator_types(p, "genetic.operator.crossover", "crossover_types", "")
    else:
        crossover_types = [
            CrossoverOp.K_POINT,
            CrossoverOp.SCATTERED,
            CrossoverOp.UNIFORM,
            CrossoverOp.ROUND_TYPE_CROSSOVER,
            CrossoverOp.BEST_TEAM_CROSSOVER,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "crossover_types", crossover_types)

    if p.has_option("genetic.operator.crossover", "crossover_ks"):
        crossover_ks = _parse_operator_types(p, "genetic.operator.crossover", "crossover_ks", "", "int")
    else:
        crossover_ks = [
            1,
            2,
            4,
            8,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "crossover_ks", crossover_ks)

    if p.has_option("genetic.operator.mutation", "mutation_types"):
        mutation_types = _parse_operator_types(p, "genetic.operator.mutation", "mutation_types", "")
    else:
        mutation_types = [
            MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION,
            MutationOp.SWAP_MATCH_SAME_LOCATION,
            MutationOp.SWAP_MATCH_SAME_TIME,
            MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            MutationOp.SWAP_TEAM_SAME_LOCATION,
            MutationOp.SWAP_TEAM_SAME_TIME,
            MutationOp.SWAP_TABLE_SIDE,
            MutationOp.INVERSION,
            MutationOp.SCRAMBLE,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "mutation_types", mutation_types)

    return OperatorConfig(selection_types, crossover_types, crossover_ks, mutation_types)


def load_ga_parameters(args: Namespace, p: ConfigParser) -> GaParameters:
    """Build a GaParameters, overriding defaults with any provided CLI args.

    Args:
        args (Namespace): Parsed command-line arguments.
        p (ConfigParser): Configuration parser with default values.

    Returns:
        GaParameters: Parameters for the genetic algorithm.

    """
    config_genetic = p["genetic"] if p.has_section("genetic") else {}

    if not config_genetic:
        msg = "No 'genetic' section found in the configuration file."
        raise KeyError(msg)

    params = {
        "population_size": config_genetic.getint("population_size", 16),
        "generations": config_genetic.getint("generations", 128),
        "offspring_size": config_genetic.getint("offspring_size", 12),
        "selection_size": config_genetic.getint("selection_size", 6),
        "crossover_chance": config_genetic.getfloat("crossover_chance", 0.5),
        "mutation_chance": config_genetic.getfloat("mutation_chance", 0.5),
        "num_islands": config_genetic.getint("num_islands", 10),
        "migration_interval": config_genetic.getint("migration_interval", 10),
        "migration_size": config_genetic.getint("migration_size", 4),
    }

    for key in params:
        if cli_val := getattr(args, key, None):
            params[key] = cli_val

    return GaParameters(**params)


def load_rng(args: Namespace, p: ConfigParser) -> Random:
    """Set up the random number generator."""
    rng_seed = ""

    if args.rng_seed is not None:
        rng_seed = args.rng_seed
    elif "genetic" in p and "seed" in p["genetic"]:
        rng_seed = p["genetic"]["seed"].strip()

    if not rng_seed:
        rng_seed = Random().randint(*RANDOM_SEED_RANGE)

    logger.debug("Using RNG seed: %d", rng_seed)
    return Random(rng_seed)


def _get_config_parser(path: Path | None = None) -> ConfigParser:
    """Get a ConfigParser instance for the given config file path.

    Args:
        path (Path | None): Path to the configuration file.

    Returns:
        ConfigParser: The configured ConfigParser instance.

    """
    if path is None:
        try:
            path = Path("fll_scheduler_ga/config.ini").resolve()
        except FileNotFoundError:
            logger.exception("Configuration file not found. Please provide a valid path.")

    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    logger.debug("Configuration file loaded from %s", path)
    return parser


def _parse_teams_config(p: ConfigParser) -> tuple[int, dict[int, int | str]]:
    """Parse and return a list of team IDs from the configuration."""
    if not p.has_section("teams"):
        msg = "No 'teams' section found in the configuration file."
        raise KeyError(msg)

    has_num = p.has_option("teams", "num_teams")
    has_ids = p.has_option("teams", "identities")

    if not (has_num or has_ids):
        msg = "No 'identities' or 'num_teams' option found in the 'teams' section."
        raise KeyError(msg)

    p_teams = p["teams"]

    num_teams = p_teams.getint("num_teams") if has_num else 0
    identities = [i.strip() for i in p_teams.get("identities", "").split(",") if i.strip()] if has_ids else []

    if has_num and not identities:
        identities = [str(i) for i in range(1, num_teams + 1)]

    if not has_num and identities:
        num_teams = len(identities)

    if (has_num and has_ids) and num_teams != len(identities):
        msg = "Number of teams does not match number of identities."
        raise ValueError(msg)

    team_ids = {
        i: int(team_id) if team_id.isdigit() else team_id
        for i, team_id in enumerate(
            identities,
            start=1,
        )
    }
    return num_teams, team_ids


def _parse_time_config(p: ConfigParser) -> str:
    """Parse and return the time format configuration value."""
    if not p.has_section("time"):
        msg = "No 'time' section found in the configuration file."
        raise KeyError(msg)

    if not p.has_option("time", "format"):
        msg = "No 'format' option found in the 'time' section."
        raise KeyError(msg)

    fmt_val = p["time"].getint("format", 24)
    time_fmt = {
        12: "%I:%M %p",
        24: "%H:%M",
    }.get(fmt_val)

    if time_fmt is None:
        msg = "Invalid time format. Must be 12 or 24."
        raise ValueError(msg)

    return time_fmt


def _parse_fitness_config(p: ConfigParser) -> tuple[float, float]:
    """Parse and return fitness-related configuration values."""
    if not p.has_section("fitness"):
        msg = "No 'fitness' section found in the configuration file."
        raise KeyError(msg)

    pfit = p["fitness"]
    weight_floats = (
        pfit.getfloat("weight_mean", 3),
        pfit.getfloat("weight_variation", 1),
    )
    weights = tuple(max(0, w) for w in weight_floats)
    total = sum(weights)
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

        p_round = p[section]
        roundtype = p_round.get("round_type")
        rounds_per_team = p_round.getint("rounds_per_team")
        teams_per_round = p_round.getint("teams_per_round")

        if start_time := p_round.get("start_time", ""):
            start_time = datetime.strptime(start_time.strip(), time_fmt).replace(tzinfo=UTC)

        if stop_time := p_round.get("stop_time", ""):
            stop_time = datetime.strptime(stop_time.strip(), time_fmt).replace(tzinfo=UTC)

        duration_minutes = timedelta(minutes=p_round.getint("duration_minutes"))

        if times := p_round.get("times", []):
            times = [datetime.strptime(t.strip(), time_fmt).replace(tzinfo=UTC) for t in times.split(",")]
            start_time = times[0]

        location = p_round.get("location")
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


def _validate_round_section(p: ConfigParser, sect: str) -> None:
    """Validate round sections in config file."""
    if not p.has_section(sect):
        msg = f"Missing section: '{sect}'"
        raise KeyError(msg)

    options = ("round_type", "rounds_per_team", "teams_per_round", "duration_minutes", "location")
    for option in options:
        if not p[sect].get(option):
            msg = f"No '{option}' option found in section '{sect}'."
            raise KeyError(msg)

    if not p[sect].get("start_time") and not p[sect].get("times"):
        msg = f"Either 'start_time' or 'times' must be specified in section '{sect}'."
        raise KeyError(msg)


def _parse_operator_types(p: ConfigParser, section: str, option: str, fallback: str, dtype: str = "") -> list[str]:
    """Parse a list of operator types from the configuration."""
    if dtype == "int":
        return [int(i.strip()) for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]
    return [i.strip() for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]


def load_location_config(p: ConfigParser) -> dict[tuple[str, int | str, int, int], Location]:
    """Parse and return a dictionary of location IDs to names from the configuration."""
    lconfig = {}
    for s in (s for s in p.sections() if s.startswith("location")):
        lconfig[s] = dict(p[s].items())

    locations = {}
    for data in lconfig.values():
        name = data["name"]
        sides = int(data["sides"])
        teams_per_round = int(data["teams_per_round"])

        for i in (i.strip() for i in data["identities"].split(",") if i.strip()):
            identity = int(i) if i.isdigit() else i

            for j in range(1, sides + 1):
                side = 0 if sides == 1 else j
                loc_key = (name, identity, teams_per_round, side)
                loc_obj = Location(
                    name=name,
                    identity=identity,
                    teams_per_round=teams_per_round,
                    side=side,
                )
                locations[loc_key] = loc_obj

    return locations
