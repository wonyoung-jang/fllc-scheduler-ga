"""Configuration for the FLL Scheduler GA application."""

import argparse
import logging
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random

from ..data_model.event import Round
from ..data_model.location import Location
from ..data_model.schedule import Schedule
from .config import RoundType, TournamentConfig
from .constants import HHMM_FMT, RANDOM_SEED_RANGE, CrossoverOp, MutationOp, SelectionOp
from .ga_operators_config import OperatorConfig
from .ga_parameters import GaParameters

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppConfig:
    """Configuration for the FLL Scheduler GA application."""

    tournament: TournamentConfig
    operators: OperatorConfig
    ga_params: GaParameters
    rng: Random


def create_app_config(args: argparse.Namespace, path: Path | None = None) -> AppConfig:
    """Create and return the application configuration."""
    if path is None:
        path = Path(args.config_file).resolve()

    parser = _get_config_parser(path)
    tournament = load_tournament_config(parser)
    operators = load_operator_config(parser)
    ga_params = load_ga_parameters(args, parser)
    rng = load_rng(args, parser)
    return AppConfig(tournament, operators, ga_params, rng)


def load_tournament_config(parser: ConfigParser) -> TournamentConfig:
    """Load and return the tournament configuration from the provided ConfigParser."""
    num_teams, team_ids = _parse_teams_config(parser)
    parsed_rounds, round_reqs = _parse_rounds_config(parser, num_teams)
    all_rounds_per_team = [r.rounds_per_team for r in parsed_rounds]
    total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
    unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1
    weight_mean = parser.get("fitness", "weight_mean", fallback="3")
    weight_variation = parser.get("fitness", "weight_variation", fallback="1")
    weight_range = parser.get("fitness", "weight_range", fallback="1")
    weight_floats = tuple(max(0, float(w)) for w in (weight_mean, weight_variation, weight_range))
    weight_sum = sum(w for w in weight_floats)
    weights = tuple(w / weight_sum for w in weight_floats)
    Schedule.team_identities = team_ids
    return TournamentConfig(
        num_teams,
        parsed_rounds,
        round_reqs,
        total_slots,
        unique_opponents_possible,
        weights,
    )


def load_operator_config(parser: ConfigParser) -> OperatorConfig:
    """Parse and return the operator configuration from the provided ConfigParser.

    Args:
        parser (ConfigParser): The ConfigParser instance with operator configuration.

    Returns:
        OperatorConfig: The parsed operator configuration.

    """
    if parser.has_option("genetic.operator.selection", "selection_types"):
        selection_types = _parse_operator_types(parser, "genetic.operator.selection", "selection_types", "")
    else:
        selection_types = [SelectionOp.TOURNAMENT_SELECT, SelectionOp.RANDOM_SELECT]
        logger.warning("%s not found in config. Using defaults: %s", "selection_types", selection_types)

    if parser.has_option("genetic.operator.crossover", "crossover_types"):
        crossover_types = _parse_operator_types(parser, "genetic.operator.crossover", "crossover_types", "")
    else:
        crossover_types = [
            CrossoverOp.K_POINT,
            CrossoverOp.SCATTERED,
            CrossoverOp.UNIFORM,
            CrossoverOp.ROUND_TYPE_CROSSOVER,
            CrossoverOp.PARTIAL_CROSSOVER,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "crossover_types", crossover_types)

    if parser.has_option("genetic.operator.crossover", "crossover_ks"):
        crossover_ks = _parse_operator_types(parser, "genetic.operator.crossover", "crossover_ks", "", "int")
    else:
        crossover_ks = [1, 2, 3]
        logger.warning("%s not found in config. Using defaults: %s", "crossover_ks", crossover_ks)

    if parser.has_option("genetic.operator.mutation", "mutation_types"):
        mutation_types = _parse_operator_types(parser, "genetic.operator.mutation", "mutation_types", "")
    else:
        mutation_types = [
            MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION,
            MutationOp.SWAP_MATCH_SAME_LOCATION,
            MutationOp.SWAP_MATCH_SAME_TIME,
            MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            MutationOp.SWAP_TEAM_SAME_LOCATION,
            MutationOp.SWAP_TEAM_SAME_TIME,
            MutationOp.SWAP_TABLE_SIDE,
        ]
        logger.warning("%s not found in config. Using defaults: %s", "mutation_types", mutation_types)

    return OperatorConfig(selection_types, crossover_types, crossover_ks, mutation_types)


def load_ga_parameters(args: argparse.Namespace, config_parser: ConfigParser) -> GaParameters:
    """Build a GaParameters, overriding defaults with any provided CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config_parser (ConfigParser): Configuration parser with default values.

    Returns:
        GaParameters: Parameters for the genetic algorithm.

    """
    config_genetic = config_parser["genetic"] if config_parser.has_section("genetic") else {}

    if not config_genetic:
        msg = "No 'genetic' section found in the configuration file. Using default values."
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


def load_rng(args: argparse.Namespace, config_parser: ConfigParser) -> Random:
    """Set up the random number generator."""
    rng_seed = ""

    if args.rng_seed is not None:
        rng_seed = args.rng_seed
    elif "genetic" in config_parser and "seed" in config_parser["genetic"]:
        rng_seed = config_parser["genetic"]["seed"].strip()

    if not rng_seed:
        rng_seed = Random().randint(*RANDOM_SEED_RANGE)

    logger.info("Using RNG seed: %d", rng_seed)
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


def _parse_teams_config(parser: ConfigParser) -> tuple[int, dict[int, int | str]]:
    """Parse and return a list of team IDs from the configuration."""
    if not parser.has_section("teams"):
        msg = "No 'teams' section found in the configuration file."
        raise KeyError(msg)

    has_num = parser.has_option("teams", "num_teams")
    has_ids = parser.has_option("teams", "identities")

    if not (has_num or has_ids):
        msg = "No 'identities' or 'num_teams' option found in the 'teams' section."
        raise KeyError(msg)

    p_teams = parser["teams"]

    num_teams = p_teams.getint("num_teams") if has_num else 0
    identities = [i.strip() for i in p_teams.get("identities", "").split(",") if i.strip()] if has_ids else []

    if has_num and not identities:
        identities = [str(i) for i in range(1, num_teams + 1)]

    if not has_num and identities:
        num_teams = len(identities)

    if (has_num and has_ids) and num_teams != len(identities):
        msg = "Number of teams does not match number of identities."
        raise ValueError(msg)

    team_ids = {}
    for i, team_id in enumerate(identities, start=1):
        team_ids[i] = int(team_id) if team_id.isdigit() else team_id
    return num_teams, team_ids


def _parse_rounds_config(parser: ConfigParser, num_teams: int) -> tuple[list[Round], dict[RoundType, int]]:
    """Parse and return a list of Round objects from the configuration.

    Args:
        parser (ConfigParser): The ConfigParser instance with tournament configuration.
        num_teams (int): The total number of teams in the tournament.

    Returns:
        list[Round]: A list of Round objects parsed from the configuration.
        dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.

    """
    rounds: list[Round] = []
    round_reqs = {}
    all_locations = load_location_config(parser)
    round_sections = (s for s in parser.sections() if s.startswith("round"))

    for section in round_sections:
        p_section = parser[section]
        roundtype = p_section.get("round_type")
        rounds_per_team = p_section.getint("rounds_per_team")
        round_reqs[roundtype] = rounds_per_team

        if start_time := p_section.get("start_time", ""):
            start_time = datetime.strptime(start_time, HHMM_FMT).replace(tzinfo=UTC)

        if stop_time := p_section.get("stop_time", ""):
            stop_time = datetime.strptime(stop_time, HHMM_FMT).replace(tzinfo=UTC)

        if times := p_section.get("times", []):
            times = [datetime.strptime(t.strip(), HHMM_FMT).replace(tzinfo=UTC) for t in times.split(",")]
            start_time = times[0]

        teams_per_round = p_section.getint("teams_per_round")
        duration_minutes = p_section.getint("duration_minutes")
        duration_minutes = timedelta(minutes=duration_minutes)
        location = p_section.get("location")
        locations = [loc for loc in all_locations.values() if loc.name == location]

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

    return rounds, round_reqs


def _parse_operator_types(p: ConfigParser, section: str, option: str, fallback: str, dtype: str = "") -> list[str]:
    """Parse a list of operator types from the configuration."""
    if dtype == "int":
        return [int(i.strip()) for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]
    return [i.strip() for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]


def load_location_config(parser: ConfigParser) -> dict[tuple[str, int | str, int, int], Location]:
    """Parse and return a dictionary of location IDs to names from the configuration."""
    _lconfig = {}
    for s in (s for s in parser.sections() if s.startswith("location")):
        _lconfig[s] = {}
        for k, v in parser[s].items():
            _lconfig[s][k] = v

    locations = {}
    for data in _lconfig.values():
        name = data["name"]
        sides = int(data["sides"])
        teams_per_round = int(data["teams_per_round"])

        for i in (i.strip() for i in data["identities"].split(",") if i.strip()):
            oneside = sides == 1
            isdigit = i.isdigit()
            identity = int(i) if isdigit else i

            for j in range(1, sides + 1):
                side = 0 if oneside else j
                loc_key = (name, identity, teams_per_round, side)
                loc_obj = Location(
                    name=name,
                    identity=identity,
                    teams_per_round=teams_per_round,
                    side=side,
                )
                locations[loc_key] = loc_obj

    return locations
