"""Configuration for the FLL Scheduler GA application."""

import argparse
import logging
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random

from ..data_model.event import Round
from .config import RoundType, TournamentConfig
from .constants import HHMM_FMT, RANDOM_SEED_RANGE, CrossoverOps, MutationOps, SelectionOps
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
    parsed_rounds, round_reqs, num_teams = _parse_rounds(parser)
    all_rounds_per_team = [r.rounds_per_team for r in parsed_rounds]
    total_slots = sum(num_teams * rpt for rpt in all_rounds_per_team)
    unique_opponents_possible = 1 <= max(all_rounds_per_team) <= num_teams - 1
    weight_mean = parser.get("DEFAULT", "weight_mean", fallback="3")
    weight_variation = parser.get("DEFAULT", "weight_variation", fallback="1")
    weight_range = parser.get("DEFAULT", "weight_range", fallback="1")
    weight_floats = tuple(max(0, float(w)) for w in (weight_mean, weight_variation, weight_range))
    weight_sum = sum(w for w in weight_floats)
    weights = tuple(w / weight_sum for w in weight_floats)
    config = TournamentConfig(
        num_teams,
        parsed_rounds,
        round_reqs,
        total_slots,
        unique_opponents_possible,
        weights,
    )
    logger.debug("Tournament configuration loaded: %s", config)
    return config


def load_operator_config(parser: ConfigParser) -> OperatorConfig:
    """Parse and return the operator configuration from the provided ConfigParser.

    Args:
        parser (ConfigParser): The ConfigParser instance with operator configuration.

    Returns:
        OperatorConfig: The parsed operator configuration.

    """
    selection_types = _parse_operator_types(parser, "genetic.operator.selection", "selection_types", "")
    crossover_types = _parse_operator_types(parser, "genetic.operator.crossover", "crossover_types", "")
    crossover_ks = _parse_operator_types(parser, "genetic.operator.crossover", "crossover_ks", "", "int")
    mutation_types = _parse_operator_types(parser, "genetic.operator.mutation", "mutation_types", "")

    if not selection_types:
        selection_types = [
            SelectionOps.TOURNAMENT_SELECT,
            SelectionOps.RANDOM_SELECT,
        ]
        logger.warning("No selection types enabled in the configuration. Using defaults: %s", selection_types)

    if not crossover_types:
        crossover_types = [
            CrossoverOps.K_POINT,
            CrossoverOps.SCATTERED,
            CrossoverOps.UNIFORM,
            CrossoverOps.ROUND_TYPE_CROSSOVER,
            CrossoverOps.PARTIAL_CROSSOVER,
        ]
        logger.warning("No crossover types enabled in the configuration. Using defaults: %s", crossover_types)

    if not crossover_ks:
        crossover_ks = [
            1,
            2,
            3,
        ]
        logger.warning("No crossover ks values provided in the configuration. Using defaults: %s", crossover_ks)

    if not mutation_types:
        mutation_types = [
            MutationOps.SWAP_MATCH_CROSS_TIME_LOCATION,
            MutationOps.SWAP_MATCH_SAME_LOCATION,
            MutationOps.SWAP_MATCH_SAME_TIME,
            MutationOps.SWAP_TEAM_CROSS_TIME_LOCATION,
            MutationOps.SWAP_TEAM_SAME_LOCATION,
            MutationOps.SWAP_TEAM_SAME_TIME,
            MutationOps.SWAP_TABLE_SIDE,
        ]
        logger.warning("No mutation types enabled in the configuration. Using defaults: %s", mutation_types)

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
        "elite_size": config_genetic.getint("elite_size", 2),
        "selection_size": config_genetic.getint("selection_size", 4),
        "crossover_chance": config_genetic.getfloat("crossover_chance", 0.5),
        "mutation_chance": config_genetic.getfloat("mutation_chance", 0.05),
        "num_islands": config_genetic.getint("num_islands", 10),
        "migration_interval": config_genetic.getint("migration_interval", 10),
        "migration_size": config_genetic.getint("migration_size", 2),
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


def _parse_rounds(parser: ConfigParser) -> tuple[list[Round], dict[RoundType, int], int]:
    """Parse and return a list of Round objects from the configuration.

    Args:
        parser (ConfigParser): The ConfigParser instance with tournament configuration.

    Returns:
        list[Round]: A list of Round objects parsed from the configuration.
        dict[RoundType, int]: A dictionary mapping round types to the number of rounds per team.
        int: The total number of teams in the tournament.

    """
    num_teams = parser["DEFAULT"].getint("num_teams")
    rounds: list[Round] = []
    round_reqs = {}

    for section in parser.sections():
        if not section.startswith("round"):
            continue

        round_type = parser[section].get("round_type")
        rounds_per_team = parser[section].getint("rounds_per_team")
        round_reqs[round_type] = rounds_per_team

        if start_time := parser[section].get("start_time", ""):
            start_time = datetime.strptime(start_time, HHMM_FMT).replace(tzinfo=UTC)

        if stop_time := parser[section].get("stop_time", ""):
            stop_time = datetime.strptime(stop_time, HHMM_FMT).replace(tzinfo=UTC)

        if times := parser[section].get("times", []):
            times = [datetime.strptime(t.strip(), HHMM_FMT).replace(tzinfo=UTC) for t in times.split(",")]
            start_time = times[0] if times else start_time

        teams_per_round = parser[section].getint("teams_per_round")
        duration_minutes = timedelta(minutes=parser[section].getint("duration_minutes"))
        num_locations = parser[section].getint("num_locations")

        rounds.append(
            Round(
                round_type,
                rounds_per_team,
                teams_per_round,
                times,
                start_time,
                stop_time,
                duration_minutes,
                num_locations,
                num_teams,
            )
        )

    if not rounds:
        msg = "No rounds defined in the configuration file."
        raise ValueError(msg)

    return rounds, round_reqs, num_teams


def _parse_operator_types(p: ConfigParser, section: str, option: str, fallback: str, dtype: str = "") -> list[str]:
    """Parse a list of operator types from the configuration."""
    if dtype == "int":
        return [int(i.strip()) for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]
    return [i.strip() for i in p.get(section, option, fallback=fallback).split(",") if i.strip()]
