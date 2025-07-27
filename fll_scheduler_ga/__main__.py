"""Main entry point for the fll-scheduler-ga package."""

import argparse
import logging
import shelve
from configparser import ConfigParser
from importlib.metadata import version
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.benchmark import FitnessBenchmark
from fll_scheduler_ga.config.config import OperatorConfig, TournamentConfig, load_tournament_config
from fll_scheduler_ga.data_model.event import EventConflicts, EventFactory
from fll_scheduler_ga.data_model.team import TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA, RANDOM_SEED_RANGE
from fll_scheduler_ga.genetic.ga_context import GaContext
from fll_scheduler_ga.genetic.ga_parameters import GaParameters
from fll_scheduler_ga.io.export import generate_summary, generate_summary_report
from fll_scheduler_ga.io.importer import CsvImporter
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.crossover import build_crossovers
from fll_scheduler_ga.operators.mutation import build_mutations
from fll_scheduler_ga.operators.nsga3 import NSGA3
from fll_scheduler_ga.operators.repairer import Repairer
from fll_scheduler_ga.operators.selection import build_selections
from fll_scheduler_ga.preflight.preflight import run_preflight_checks

logger = logging.getLogger(__name__)


def setup_environment() -> tuple[argparse.Namespace, GA, TournamentConfig]:
    """Set up the environment for the application."""
    try:
        args = _create_parser().parse_args()
        _initialize_logging(args)
        config, parser, operator_config = load_tournament_config(Path(args.config_file))
        event_factory = EventFactory(config)
        run_preflight_checks(config, event_factory)
        ga_params = _build_ga_parameters_from_args(args, parser)
        rng = _setup_rng(args, parser)
        ga_context = _create_ga_context(config, operator_config, event_factory, ga_params, rng)
        handle_seed_file(args, config, event_factory, ga_context)
        ga = _create_ga_instance(ga_context, rng)
        ga.set_seed_file(args.seed_file)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return args, ga, config


def handle_seed_file(
    args: argparse.Namespace,
    config: TournamentConfig,
    event_factory: EventFactory,
    ga_context: GaContext,
) -> None:
    """Handle the seed file for the genetic algorithm."""
    seed_path = Path(args.seed_file).resolve()
    if args.flush and seed_path.exists():
        seed_path.unlink()

    if args.import_file:
        schedule_csv_path = Path(args.import_file).resolve()
        csv_import = CsvImporter(schedule_csv_path, config, event_factory)
        if import_fitness := ga_context.evaluator.evaluate(csv_import.schedule):
            csv_import.schedule.fitness = import_fitness
            parent_dir = schedule_csv_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            generate_summary_report(
                csv_import.schedule,
                ga_context.evaluator.objectives,
                report_path,
            )

        if args.add_import_to_population:
            try:
                with shelve.open(seed_path) as shelf:
                    population = shelf.get("population", [])
                    if csv_import.schedule not in population:
                        population.append(csv_import.schedule)
                    shelf["population"] = population
                    shelf["config"] = config
            except Exception:
                logger.exception("Error loading seed file")


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
        "seed_file": "fllc_genetic",
    }
    parser = argparse.ArgumentParser(
        description="Generate a tournament schedule using a Genetic Algorithm.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('fll-scheduler-ga')}",
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
        "--rng_seed",
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
        "--mutation_chance",
        type=float,
        help="(OPTIONAL) Mutation chance (0.0 to 1.0).",
    )
    parser.add_argument(
        "--no_plotting",
        action="store_true",
        default=_default_values["no_plotting"],
        help="Disable plotting of results.",
    )

    # Seed file parameters
    seed_group = parser.add_argument_group("Seed File Parameters")
    seed_group.add_argument(
        "--seed_file",
        type=str,
        default=_default_values["seed_file"],
        help="Path to the seed file for the genetic algorithm.",
    )
    seed_group.add_argument(
        "--flush",
        action="store_true",
        help="Flush the cached population to the seed file at the end of the run.",
    )

    # Island model parameters
    island_group = parser.add_argument_group("Island Model Parameters")
    island_group.add_argument(
        "--num_islands",
        type=int,
        help="(OPTIONAL) Number of islands for the GA (enables island model if > 1).",
    )
    island_group.add_argument(
        "--migration_interval",
        type=int,
        help="(OPTIONAL) Generations between island migrations.",
    )
    island_group.add_argument(
        "--migration_size",
        type=int,
        help="(OPTIONAL) Number of individuals to migrate.",
    )

    # Schedule importer parameters
    import_group = parser.add_argument_group("Schedule Importer Parameters")
    import_group.add_argument(
        "--import_file",
        type=str,
        help="(OPTIONAL) Path to a CSV file to import a schedule.",
    )
    import_group.add_argument(
        "--add_import_to_population",
        action="store_true",
        help="(OPTIONAL) Add imported schedule to the initial population.",
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
    logger.debug("Starting FLLC Scheduler with args:\n%s", args_str)


def _build_ga_parameters_from_args(args: argparse.Namespace, config_parser: ConfigParser) -> GaParameters:
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


def _setup_rng(args: argparse.Namespace, config_parser: ConfigParser) -> Random:
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


def _create_ga_context(
    config: TournamentConfig,
    operator_config: OperatorConfig,
    event_factory: EventFactory,
    ga_params: GaParameters,
    rng: Random,
) -> GaContext:
    """Create and return a GaContext with the provided configuration."""
    event_conflicts = EventConflicts(event_factory)
    team_factory = TeamFactory(config, event_conflicts.conflicts)
    repairer = Repairer(rng, config, event_factory)
    selections = tuple(build_selections(operator_config, rng, ga_params))
    crossovers = tuple(build_crossovers(operator_config, rng, team_factory, event_factory))
    mutations = tuple(build_mutations(operator_config, rng))
    benchmark = FitnessBenchmark(config, event_factory)
    evaluator = FitnessEvaluator(config, benchmark)
    return GaContext(
        config=config,
        ga_params=ga_params,
        event_factory=event_factory,
        team_factory=team_factory,
        repairer=repairer,
        evaluator=evaluator,
        nsga3=NSGA3(rng, len(evaluator.objectives), ga_params.population_size),
        logger=logger,
        selections=selections,
        crossovers=crossovers,
        mutations=mutations,
    )


def _create_ga_instance(context: GaContext, rng: Random) -> GA:
    """Create and return a GA instance with the provided configuration."""
    return GA(
        context=context,
        rng=rng,
        observers=(
            LoggingObserver(logger),
            TqdmObserver(),
        ),
    )


def save_population_to_seed_file(
    ga: GA,
    config: TournamentConfig,
    seed_file: str | Path,
    *,
    front: bool = False,
) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    population = ga.pareto_front() if front else ga.total_population

    if not population:
        logger.warning("No population to save to seed file.")
        return

    path = Path(seed_file)
    logger.info("Saving final population of size %d to seed file: %s", len(population), path)
    try:
        with shelve.open(path) as shelf:
            shelf["population"] = population
            shelf["config"] = config
    except OSError:
        logger.exception("Error saving population to seed file: %s", path)


def main() -> None:
    """Run the fll-scheduler-ga application."""
    env = setup_environment()
    if not env:
        logger.error("Failed to set up the environment. Exiting.")
        return

    args, ga, config = env

    try:
        ga.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Saving final population and results before exiting.")
    except Exception:
        logger.exception("An unhandled error occurred during the GA run. Saving state before exiting.")
    finally:
        if ga.total_population:  # only save if a final population exists
            save_population_to_seed_file(ga, config, args.seed_file, front=True)
            generate_summary(args, ga)

    logger.info("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
