"""Main entry point for the fll-scheduler-ga package."""

import argparse
import logging
from configparser import ConfigParser
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.config import get_config_parser, load_tournament_config
from fll_scheduler_ga.data_model.event import EventConflicts, EventFactory
from fll_scheduler_ga.data_model.team import TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA, RANDOM_SEED
from fll_scheduler_ga.genetic.ga_parameters import GaParameters
from fll_scheduler_ga.io.export import generate_summary
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.crossover import KPoint, RoundTypeCrossover, Scattered, Uniform
from fll_scheduler_ga.operators.mutation import (
    SwapMatchMutation,
    SwapTeamMutation,
)
from fll_scheduler_ga.operators.repairer import Repairer
from fll_scheduler_ga.operators.selection import (
    Elitism,
    TournamentSelect,
)
from fll_scheduler_ga.preflight.preflight import run_preflight_checks

logger = logging.getLogger(__name__)


def setup_environment() -> tuple[argparse.Namespace, GA]:
    """Set up the environment for the application."""
    try:
        args = _create_parser().parse_args()
        _initialize_logging(args)
        config_parser = get_config_parser(Path(args.config_file))
        config = load_tournament_config(config_parser)
        event_factory = EventFactory(config)
        run_preflight_checks(config, event_factory)
        ga_params = _build_ga_parameters_from_args(args, config_parser)
        rng = _setup_rng(args, config_parser)
        ga = _create_ga_instance(config, event_factory, ga_params, rng)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return args, ga


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the application.

    Returns:
        argparse.ArgumentParser: Configured argument parser.

    """
    _default_values = {
        "config_file": "fll_scheduler_ga/config.ini",
        "output_dir": "fllc_schedule_outputs",
        "log_file": "fll_scheduler_ga.log",
        "loglevel_file": "DEBUG",
        "loglevel_console": "INFO",
        "no_plotting": False,
    }
    parser = argparse.ArgumentParser(
        description="Generate a tournament schedule using a Genetic Algorithm.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=_default_values["config_file"],
        help="Path to config .ini file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_default_values["output_dir"],
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=_default_values["log_file"],
        help="Path to the log file.",
    )
    parser.add_argument(
        "--loglevel_file",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["loglevel_file"],
        help="Logging level for the file output.",
    )
    parser.add_argument(
        "--loglevel_console",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["loglevel_console"],
        help="Logging level for the console output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(OPTIONAL) Random seed for reproducibility.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        help="(OPTIONAL) Population size for the GA.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="(OPTIONAL) Number of generations to run.",
    )
    parser.add_argument(
        "--elite_size",
        type=int,
        help="(OPTIONAL) Number of elite individuals.",
    )
    parser.add_argument(
        "--selection_size",
        type=int,
        help="(OPTIONAL) Size of parent selection.",
    )
    parser.add_argument(
        "--crossover_chance",
        type=float,
        help="(OPTIONAL) Chance of crossover (0.0 to 1.0).",
    )
    parser.add_argument(
        "--mutation_chance_low",
        type=float,
        help="(OPTIONAL) Lower bound of mutation chance (0.0 to 1.0).",
    )
    parser.add_argument(
        "--mutation_chance_high",
        type=float,
        help="(OPTIONAL) Upper bound of mutation chance (0.0 to 1.0).",
    )
    parser.add_argument(
        "--no_plotting",
        action="store_true",
        default=_default_values["no_plotting"],
        help="Disable plotting of results.",
    )
    return parser


def _initialize_logging(args: argparse.Namespace) -> None:
    """Initialize logging for the application."""
    file_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(args.loglevel_file)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(name)s] %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.loglevel_console)
    console_handler.setFormatter(logging.Formatter("%(levelname)s[%(name)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    args_str = "\n".join(f"\t{k} = {v}" for k, v in args.__dict__.items())
    logger.info("Starting fll-scheduler-ga application with args:\n%s", args_str)


def _build_ga_parameters_from_args(args: argparse.Namespace, config_parser: ConfigParser) -> GaParameters:
    """Build a GaParameters, overriding defaults with any provided CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config_parser (ConfigParser): Configuration parser with default values.

    Returns:
        GaParameters: Parameters for the genetic algorithm.

    """
    config_genetic = config_parser["genetic"]
    config_args = {
        "population_size": config_genetic.getint("population_size", 16),
        "generations": config_genetic.getint("generations", 128),
        "elite_size": config_genetic.getint("elite_size", 2),
        "selection_size": config_genetic.getint("selection_size", 4),
        "crossover_chance": config_genetic.getfloat("crossover_chance", 0.5),
        "mutation_chance_low": config_genetic.getfloat("mutation_chance_low", 0.2),
        "mutation_chance_high": config_genetic.getfloat("mutation_chance_high", 0.8),
    }
    cli_args = {
        "population_size": args.population_size,
        "generations": args.generations,
        "elite_size": args.elite_size,
        "selection_size": args.selection_size,
        "crossover_chance": args.crossover_chance,
        "mutation_chance_low": args.mutation_chance_low,
        "mutation_chance_high": args.mutation_chance_high,
    }
    provided_args = {k: v if v is not None else config_args[k] for k, v in cli_args.items()}
    ga_params = GaParameters(**provided_args)
    logger.info("Using GA parameters: %s", ga_params)
    return ga_params


def _setup_rng(args: argparse.Namespace, config_parser: ConfigParser) -> Random:
    """Set up the random number generator."""
    config_genetic = config_parser["genetic"]
    if "seed" in config_genetic:
        rng_seed = config_genetic.get("seed", 0)
        logger.info("Using RNG seed from config: %d", rng_seed)
    elif args.seed is not None:
        rng_seed = args.seed
        logger.info("Using provided RNG seed: %d", args.seed)
    else:
        rng_seed = Random().randint(*RANDOM_SEED)
        logger.info("Using master RNG seed: %d", rng_seed)
    return Random(rng_seed)


def _create_ga_instance(config: dict, event_factory: EventFactory, ga_params: GaParameters, rng: Random) -> GA:
    """Create and return a GA instance with the provided configuration."""
    event_conflicts = EventConflicts(event_factory)
    team_factory = TeamFactory(config, event_conflicts.conflicts)
    selection = TournamentSelect(rng, tournament_size=ga_params.selection_size)
    elitism = Elitism(rng)  # Separate survival selection
    evaluator = FitnessEvaluator(config)
    observers = (
        LoggingObserver(logger),
        TqdmObserver(logger),
    )

    events_list = event_factory.flat_list()
    repairer = Repairer(rng, config, set(events_list))

    crossovers = (
        KPoint(team_factory, events_list, rng, repairer, k=1),  # Single-point
        KPoint(team_factory, events_list, rng, repairer, k=2),  # Two-point
        KPoint(team_factory, events_list, rng, repairer, k=8),  # K point (8)
        Scattered(team_factory, events_list, rng, repairer),
        Uniform(team_factory, events_list, rng, repairer),
        RoundTypeCrossover(team_factory, events_list, rng, repairer),
    )

    mutations = (
        SwapMatchMutation(rng, same_timeslot=False, same_location=False),  # Across timeslots and locations
        SwapMatchMutation(rng, same_timeslot=False, same_location=True),  # Across timeslots, same location
        SwapMatchMutation(rng, same_timeslot=True, same_location=False),  # Same timeslot, across locations
        SwapTeamMutation(rng, same_timeslot=False, same_location=False),  # Across timeslots and locations
        SwapTeamMutation(rng, same_timeslot=False, same_location=True),  # Across timeslots, same location
        SwapTeamMutation(rng, same_timeslot=True, same_location=False),  # Same timeslot, across locations
    )

    return GA(
        ga_params=ga_params,
        config=config,
        rng=rng,
        event_factory=event_factory,
        team_factory=team_factory,
        selection=selection,
        elitism=elitism,
        crossovers=crossovers,
        mutations=mutations,
        logger=logger,
        observers=observers,
        evaluator=evaluator,
        repairer=repairer,
    )


def main() -> None:
    """Run the fll-scheduler-ga application."""
    args, ga = setup_environment()

    if ga.run():
        generate_summary(args, ga)
    else:
        logger.warning("Genetic algorithm did not produce a valid final schedule.")

    logger.info("fll-scheduler-ga application finished")


if __name__ == "__main__":
    main()
