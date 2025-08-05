"""Main entry point for the fll-scheduler-ga package."""

import pickle
from argparse import ArgumentParser, Namespace
from importlib.metadata import version
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.app_config import AppConfig, create_app_config
from fll_scheduler_ga.config.benchmark import FitnessBenchmark
from fll_scheduler_ga.config.preflight import run_preflight_checks
from fll_scheduler_ga.data_model.event import EventFactory
from fll_scheduler_ga.data_model.team import TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.genetic.ga_context import GaContext
from fll_scheduler_ga.io.export import generate_summary, generate_summary_report
from fll_scheduler_ga.io.importer import CsvImporter
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.nsga3 import NSGA3
from fll_scheduler_ga.operators.repairer import Repairer

logger = getLogger(__name__)


def setup_environment() -> tuple[Namespace, GA]:
    """Set up the environment for the application."""
    args = create_parser().parse_args()
    initialize_logging(args)

    try:
        app_config = create_app_config(args)
        ga_context = create_ga_context(app_config)

        run_preflight_checks(app_config.tournament, ga_context.event_factory)
        handle_seed_file(args, ga_context)

        ga = create_ga_instance(ga_context, app_config.rng)
        ga.set_seed_file(args.seed_file)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return args, ga


def handle_seed_file(args: Namespace, ga_context: GaContext) -> None:
    """Handle the seed file for the genetic algorithm."""
    config = ga_context.config
    path = Path(args.seed_file).resolve()

    if args.flush and path.exists():
        path.unlink(missing_ok=True)
    path.touch(exist_ok=True)

    if args.import_file:
        schedule_csv_path = Path(args.import_file).resolve()
        csv_import = CsvImporter(schedule_csv_path, config, ga_context.event_factory)
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
            population = []
            try:
                with path.open("rb") as f:
                    seed_data = pickle.load(f)
                    population.extend(seed_data.get("population", []))
            except (OSError, pickle.PicklingError):
                logger.exception("Error loading seed file")
            except EOFError:
                logger.debug("Pickle file is empty")

            if csv_import.schedule not in population:
                population.append(csv_import.schedule)

            try:
                with path.open("wb") as f:
                    data_to_cache = {
                        "population": population,
                        "config": ga_context.config,
                    }
                    pickle.dump(data_to_cache, f)
            except (OSError, pickle.PicklingError):
                logger.exception("Error loading seed file")
            except EOFError:
                logger.debug("Pickle file is empty")


def create_parser() -> ArgumentParser:
    """Create the argument parser for the application.

    Returns:
        ArgumentParser: Configured argument parser.

    """
    _default_values = {
        "config_file": "fll_scheduler_ga/config.ini",
        "output_dir": "fllc_schedule_outputs",
        "log_file": "fll_scheduler_ga.log",
        "loglevel_file": "DEBUG",
        "loglevel_console": "INFO",
        "no_plotting": False,
        "seed_file": "fllc_genetic.pkl",
        "front_only": True,
    }
    parser = ArgumentParser(
        description="Generate a tournament schedule using a Genetic Algorithm.",
    )

    # General parameters
    general_group = parser.add_argument_group("General Parameters")
    general_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('fll-scheduler-ga')}",
    )
    general_group.add_argument(
        "--config_file",
        type=str,
        default=_default_values["config_file"],
        help="Path to config .ini file.",
    )
    general_group.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    # Output parameters
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default=_default_values["output_dir"],
        help="Directory to save output files.",
    )
    output_group.add_argument(
        "--no_plotting",
        action="store_true",
        default=_default_values["no_plotting"],
        help="Disable plotting of results.",
    )

    # Logging parameters
    log_group = parser.add_argument_group("Logging Parameters")
    log_group.add_argument(
        "--log_file",
        type=str,
        default=_default_values["log_file"],
        help="Path to the log file.",
    )
    log_group.add_argument(
        "--loglevel_file",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["loglevel_file"],
        help="Logging level for the file output.",
    )
    log_group.add_argument(
        "--loglevel_console",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["loglevel_console"],
        help="Logging level for the console output.",
    )

    # Genetic algorithm parameters
    genetic_group = parser.add_argument_group("Genetic Algorithm Parameters")
    genetic_group.add_argument(
        "--population_size",
        type=int,
        help="Population size for the GA.",
    )
    genetic_group.add_argument(
        "--generations",
        type=int,
        help="Number of generations to run.",
    )
    genetic_group.add_argument(
        "--offspring_size",
        type=int,
        help="Number of offspring individuals per generation.",
    )
    genetic_group.add_argument(
        "--selection_size",
        type=int,
        help="Size of parent selection.",
    )
    genetic_group.add_argument(
        "--crossover_chance",
        type=float,
        help="Chance of crossover (0.0 to 1.0).",
    )
    genetic_group.add_argument(
        "--mutation_chance",
        type=float,
        help="Mutation chance (0.0 to 1.0).",
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
    seed_group.add_argument(
        "--front_only",
        action="store_true",
        default=_default_values["front_only"],
        help="Whether to save only the pareto front to the seed file. (default: True)",
    )

    # Island model parameters
    island_group = parser.add_argument_group("Island Model Parameters")
    island_group.add_argument(
        "--num_islands",
        type=int,
        help="Number of islands for the GA (enables island model if > 1).",
    )
    island_group.add_argument(
        "--migration_interval",
        type=int,
        help="Generations between island migrations.",
    )
    island_group.add_argument(
        "--migration_size",
        type=int,
        help="Number of individuals to migrate.",
    )

    # Schedule importer parameters
    import_group = parser.add_argument_group("Schedule Importer Parameters")
    import_group.add_argument(
        "--import_file",
        type=str,
        help="Path to a CSV file to import a schedule.",
    )
    import_group.add_argument(
        "--add_import_to_population",
        action="store_true",
        help="Add imported schedule to the initial population.",
    )

    return parser


def initialize_logging(args: Namespace) -> None:
    """Initialize logging for the application."""
    file_fmt = Formatter("[%(asctime)s] %(levelname)s[%(name)s] %(message)s")
    file_handler = FileHandler(
        filename=args.log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(args.loglevel_file)
    file_handler.setFormatter(file_fmt)

    console_fmt = Formatter("%(levelname)s[%(name)s] %(message)s")
    console_handler = StreamHandler()
    console_handler.setLevel(args.loglevel_console)
    console_handler.setFormatter(console_fmt)

    root_logger = getLogger()
    root_logger.setLevel(DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    args_str = "\n\t".join(f"  {k}: {v}" for k, v in args.__dict__.items())
    logger.debug("Starting FLLC Scheduler with args:\n\targparse.Namespace\n\t%s", args_str)


def create_ga_context(app_config: AppConfig) -> GaContext:
    """Create and return a GaContext with the provided configuration."""
    team_factory = TeamFactory(app_config.tournament)
    event_factory = EventFactory(app_config.tournament)

    repairer = Repairer(app_config.rng, app_config.tournament, event_factory)
    benchmark = FitnessBenchmark(app_config.tournament, event_factory)
    evaluator = FitnessEvaluator(app_config.tournament, benchmark)

    num_objectives = len(evaluator.objectives)
    pop_size_ref_points = app_config.ga_params.population_size * app_config.ga_params.num_islands
    pop_size_ref_points = max(pop_size_ref_points, 32)  # Ensure at least 32 reference points
    nsga3 = NSGA3(
        rng=app_config.rng,
        num_objectives=num_objectives,
        population_size=pop_size_ref_points,
    )

    return GaContext(
        app_config=app_config,
        event_factory=event_factory,
        team_factory=team_factory,
        repairer=repairer,
        evaluator=evaluator,
        logger=logger,
        nsga3=nsga3,
    )


def create_ga_instance(context: GaContext, rng: Random) -> GA:
    """Create and return a GA instance with the provided configuration."""
    return GA(
        context=context,
        rng=rng,
        observers=(
            LoggingObserver(logger),
            TqdmObserver(),
        ),
    )


def save_population_to_seed_file(ga: GA, seed_file: str | Path, *, front: bool = False) -> None:
    """Save the final population to a file to be used as a seed for a future run."""
    population = ga.pareto_front() if front else ga.total_population

    if not population:
        logger.warning("No population to save to seed file.")
        return

    data_to_cache = {
        "population": population,
        "config": ga.context.config,
    }

    path = Path(seed_file)
    logger.info("Saving final population of size %d to seed file: %s", len(population), path)
    try:
        with path.open("wb") as f:
            pickle.dump(data_to_cache, f)
    except (OSError, pickle.PicklingError, EOFError):
        logger.exception("Error saving population to seed file: %s", path)


def main() -> None:
    """Run the fll-scheduler-ga application."""
    env = setup_environment()
    if not env:
        logger.error("Failed to set up the environment. Exiting.")
        return

    args, ga = env

    try:
        ga.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Saving final population and results before exiting.")
    except Exception:
        logger.exception("An unhandled error occurred during the GA run. Saving state before exiting.")
    finally:
        if ga.total_population:  # only save if a final population exists
            save_population_to_seed_file(ga, args.seed_file, front=args.front_only)
            generate_summary(args, ga)

    logger.info("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
