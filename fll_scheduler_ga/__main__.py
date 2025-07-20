"""Main entry point for the fll-scheduler-ga package."""

import argparse
import logging
import pickle
from configparser import ConfigParser
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.config import TournamentConfig, get_config_parser, load_tournament_config
from fll_scheduler_ga.data_model.event import EventConflicts, EventFactory
from fll_scheduler_ga.data_model.team import TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA, RANDOM_SEED
from fll_scheduler_ga.genetic.ga_parameters import GaParameters
from fll_scheduler_ga.io.export import generate_summary
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.crossover import build_crossovers
from fll_scheduler_ga.operators.mutation import build_mutations
from fll_scheduler_ga.operators.repairer import Repairer
from fll_scheduler_ga.operators.selection import Elitism, build_selections
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
        ga.set_seed_file(args.seed_file)
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
        "seed_file": "fllc_genetic.pkl",
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
    parser.add_argument(
        "--seed_file",
        type=str,
        default=_default_values["seed_file"],
        help="Path to the seed file for the genetic algorithm.",
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
    config_genetic = config_parser["genetic"] if config_parser.has_section("genetic") else {}
    params = {
        "population_size": config_genetic.getint("population_size", 16),
        "generations": config_genetic.getint("generations", 128),
        "elite_size": config_genetic.getint("elite_size", 2),
        "selection_size": config_genetic.getint("selection_size", 4),
        "crossover_chance": config_genetic.getfloat("crossover_chance", 0.5),
        "mutation_chance_low": config_genetic.getfloat("mutation_chance_low", 0.2),
        "mutation_chance_high": config_genetic.getfloat("mutation_chance_high", 0.8),
    }

    for key in params:
        if cli_val := getattr(args, key, None):
            params[key] = cli_val

    ga_params = GaParameters(**params)
    logger.info("Using GA parameters: %s", ga_params)
    return ga_params


def _setup_rng(args: argparse.Namespace, config_parser: ConfigParser) -> Random:
    """Set up the random number generator."""
    if args.seed is not None:
        rng_seed = args.seed
    elif "genetic" in config_parser and "seed" in config_parser["genetic"]:
        rng_seed = config_parser["genetic"].getint("seed")
    else:
        rng_seed = Random().randint(*RANDOM_SEED)

    logger.info("Using master RNG seed: %d", rng_seed)
    return Random(rng_seed)


def _create_ga_instance(
    config: TournamentConfig,
    event_factory: EventFactory,
    ga_params: GaParameters,
    rng: Random,
) -> GA:
    """Create and return a GA instance with the provided configuration."""
    team_factory = TeamFactory(config, EventConflicts(event_factory).conflicts)
    repairer = Repairer(rng, config, event_factory)
    selections = tuple(build_selections(config, rng, ga_params))
    crossovers = tuple(build_crossovers(config, team_factory, event_factory, rng))
    mutations = tuple(build_mutations(config, rng))
    return GA(
        ga_params=ga_params,
        config=config,
        rng=rng,
        event_factory=event_factory,
        team_factory=team_factory,
        selections=selections,
        elitism=Elitism(rng),
        crossovers=crossovers,
        mutations=mutations,
        logger=logger,
        observers=(
            LoggingObserver(logger),
            TqdmObserver(logger),
        ),
        evaluator=FitnessEvaluator(config),
        repairer=repairer,
    )


def save_population_to_seed_file(ga: GA, seed_file: str | Path, *, front: bool = False) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    population = ga.pareto_front() if front else ga.population

    if not population:
        logger.warning("No population to save to seed file.")
        return

    path = Path(seed_file)
    logger.info("Saving final population of size %d to seed file: %s", len(population), path)
    try:
        with path.open("wb") as f:
            pickle.dump(population, f)
    except (OSError, pickle.PicklingError):
        logger.exception("Error saving population to seed file: %s", path)


def main() -> None:
    """Run the fll-scheduler-ga application."""
    args, ga = setup_environment()
    if not ga:
        return

    try:
        ga.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Saving final population and results before exiting.")
    except Exception:
        logger.exception("An unhandled error occurred during the GA run. Saving state before exiting.")
    finally:
        save_population_to_seed_file(ga, args.seed_file, front=True)
        generate_summary(args, ga)

    logger.info("fll-scheduler-ga application finished")


if __name__ == "__main__":
    main()
