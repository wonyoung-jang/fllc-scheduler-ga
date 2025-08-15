"""Main entry point for the fll-scheduler-ga package."""

import pickle
from argparse import Namespace
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.app_config import AppConfig, create_app_config
from fll_scheduler_ga.config.benchmark import FitnessBenchmark
from fll_scheduler_ga.config.cli import create_parser
from fll_scheduler_ga.config.ga_context import GaContext
from fll_scheduler_ga.config.preflight import run_preflight_checks
from fll_scheduler_ga.data_model.event import EventFactory
from fll_scheduler_ga.data_model.team import TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.io.export import generate_summary, generate_summary_report
from fll_scheduler_ga.io.importer import CsvImporter
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.crossover import build_crossovers
from fll_scheduler_ga.operators.mutation import build_mutations
from fll_scheduler_ga.operators.nsga3 import NSGA3
from fll_scheduler_ga.operators.repairer import Repairer
from fll_scheduler_ga.operators.selection import build_selections

logger = getLogger(__name__)


def setup_environment() -> tuple[Namespace, GA]:
    """Set up the environment for the application."""
    parser = create_parser()
    args = parser.parse_args()
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
    config = ga_context.app_config.tournament
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
                        "config": ga_context.app_config.tournament,
                    }
                    pickle.dump(data_to_cache, f)
            except (OSError, pickle.PicklingError):
                logger.exception("Error loading seed file")
            except EOFError:
                logger.debug("Pickle file is empty")


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
    pop_size_ref_points = max(pop_size_ref_points, 100)
    nsga3 = NSGA3(
        rng=app_config.rng,
        num_objectives=num_objectives,
        population_size=pop_size_ref_points,
    )

    selections = tuple(build_selections(app_config))
    crossovers = tuple(build_crossovers(app_config, team_factory, event_factory))
    mutations = tuple(build_mutations(app_config))

    return GaContext(
        app_config=app_config,
        event_factory=event_factory,
        team_factory=team_factory,
        repairer=repairer,
        evaluator=evaluator,
        logger=logger,
        nsga3=nsga3,
        selections=selections,
        crossovers=crossovers,
        mutations=mutations,
    )


def create_ga_instance(context: GaContext, rng: Random) -> GA:
    """Create and return a GA instance with the provided configuration."""
    return GA(
        context=context,
        rng=rng,
        observers=(
            TqdmObserver(),
            LoggingObserver(logger),
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
        "config": ga.context.app_config.tournament,
    }

    path = Path(seed_file)
    logger.debug("Saving final population of size %d to seed file: %s", len(population), path)

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

    logger.debug("FLLC Scheduler finished")


if __name__ == "__main__":
    main()
