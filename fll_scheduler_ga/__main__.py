"""Main entry point for the fll-scheduler-ga package."""

import argparse
import logging
from configparser import ConfigParser
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.config import get_config_parser, load_tournament_config
from fll_scheduler_ga.data_model.event import EventConflicts, EventFactory
from fll_scheduler_ga.data_model.team import Team, TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA, RANDOM_SEED
from fll_scheduler_ga.genetic.ga_parameters import GaParameters
from fll_scheduler_ga.genetic.schedule import Schedule
from fll_scheduler_ga.io.export import get_exporter
from fll_scheduler_ga.observers.loggers import LoggingObserver
from fll_scheduler_ga.observers.progress import TqdmObserver
from fll_scheduler_ga.operators.crossover import KPoint, Scattered, Uniform
from fll_scheduler_ga.operators.mutation import (
    SwapMatchAcrossLocation,
    SwapMatchWithinLocation,
    SwapMatchWithinTimeSlot,
    SwapTeamAcrossLocation,
    SwapTeamWithinLocation,
    SwapTeamWithinTimeSlot,
)
from fll_scheduler_ga.operators.selection import ElitismSelectionNSGA2, TournamentSelectionNSGA2
from fll_scheduler_ga.preflight.preflight import run_preflight_checks
from fll_scheduler_ga.visualize.plot import Plot

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
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
    return parser


def build_ga_parameters_from_args(args: argparse.Namespace, config_parser: ConfigParser) -> GaParameters:
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


def initialize_logging(args: argparse.Namespace) -> None:
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


def generate_summary_report(schedule: Schedule, evaluator: FitnessEvaluator, path: Path) -> None:
    """Generate a text summary report for a single schedule."""
    obj_names = evaluator.objectives
    scores = schedule.fitness
    with path.open("w", encoding="utf-8") as f:
        f.write(f"--- FLL Scheduler GA Summary Report ({id(schedule)}) ---\n\n")
        f.write("Objective Scores:\n")
        for name, score in zip(obj_names, scores, strict=False):
            f.write(f"  - {name}: {score:.4f}\n")

        f.write("\nNotes:\n")
        all_teams: list[Team] = schedule.all_teams()
        worst_team = min(all_teams, key=lambda t: t.score_break_time())
        f.write(f"  - Team with worst break time distribution: Team {worst_team.identity}\n")


def generate_pareto_summary(front: list[Schedule], evaluator: FitnessEvaluator, path: Path) -> None:
    """Generate a summary of the Pareto front."""
    schedule_enum_digits = len(str(len(front)))
    obj_names = evaluator.objectives
    front.sort(key=lambda s: (s.rank, -s.crowding))
    with path.open("w", encoding="utf-8") as f:
        f.write("Schedule, ID, Hash, Rank, Crowding, ")
        for name in obj_names:
            f.write(f"{name}, ")
        f.write("Sum\n")
        for i, schedule in enumerate(front, start=1):
            rank = schedule.rank
            crowding = schedule.crowding
            if crowding == float("inf"):
                crowding = 9.9999

            f.write(f"{i:0{schedule_enum_digits}}, {id(schedule)}, {hash(schedule)}, {rank}, {crowding:.4f}, ")
            for score in schedule.fitness:
                f.write(f"{score:.4f}, ")
            f.write(f"{sum(schedule.fitness):.4f}\n")


def summary(args: argparse.Namespace, ga: GA) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    plot = Plot(ga)
    plot.plot_fitness(save_dir=output_dir / "fitness_vs_generation.png")
    plot.plot_pareto_front(save_dir=output_dir / "pareto_front_tradeoffs.png")

    ga.population.sort(key=lambda s: (s.rank, -s.crowding))

    for i, schedule in enumerate(ga.pareto_front(), start=1):
        name = f"front_{schedule.rank}_schedule_{i}"
        suffixes = (
            "csv",
            "html",
        )
        for suffix in suffixes:
            suffix_subdir = output_dir / suffix
            suffix_subdir.mkdir(parents=True, exist_ok=True)
            output_path = suffix_subdir / name
            output_path = output_path.with_suffix(f".{suffix}")
            exporter = get_exporter(output_path)
            exporter.export(schedule, output_path)

        txt_subdir = output_dir / "txt"
        txt_subdir.mkdir(parents=True, exist_ok=True)
        generate_summary_report(schedule, ga.fitness, txt_subdir / f"{name}_summary.txt")

    generate_pareto_summary(ga.population, ga.fitness, output_dir / "pareto_summary.csv")


def setup_environment(args: argparse.Namespace) -> tuple[dict, ConfigParser, EventFactory]:
    """Set up the environment for the application."""
    try:
        config_parser = get_config_parser(Path(args.config_file))
        config = load_tournament_config(config_parser)
        event_factory = EventFactory(config)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
    else:
        return config, config_parser, event_factory


def setup_rng(args: argparse.Namespace) -> Random:
    """Set up the random number generator."""
    if args.seed is not None:
        rng_seed = args.seed
        logger.info("Using provided RNG seed: %d", args.seed)
    else:
        rng_seed = Random().randint(*RANDOM_SEED)
        logger.info("Using master RNG seed: %d", rng_seed)
    return Random(rng_seed)


def create_ga_instance(config: dict, event_factory: EventFactory, ga_parameters: GaParameters, rng: Random) -> GA:
    """Create and return a GA instance with the provided configuration."""
    event_conflicts = EventConflicts(event_factory)
    team_factory = TeamFactory(config, event_conflicts.conflicts)
    selections = (TournamentSelectionNSGA2(ga_parameters, rng),)
    elitism = ElitismSelectionNSGA2(ga_parameters, rng)

    crossovers = (
        KPoint(team_factory, event_factory, rng, k=1),  # Single-point
        KPoint(team_factory, event_factory, rng, k=2),  # Two-point
        KPoint(team_factory, event_factory, rng, k=8),  # K point (8)
        Scattered(team_factory, event_factory, rng),
        Uniform(team_factory, event_factory, rng),
    )

    mutations = (
        SwapMatchWithinTimeSlot(rng),
        SwapMatchWithinLocation(rng),
        SwapMatchAcrossLocation(rng),
        SwapTeamWithinLocation(rng),
        SwapTeamWithinTimeSlot(rng),
        SwapTeamAcrossLocation(rng),
    )

    evaluator = FitnessEvaluator(config)

    observers = [
        LoggingObserver(logger),
        TqdmObserver(logger),
    ]

    return GA(
        ga_parameters=ga_parameters,
        config=config,
        rng=rng,
        event_factory=event_factory,
        team_factory=team_factory,
        selections=selections,
        elitism=elitism,
        crossovers=crossovers,
        mutations=mutations,
        logger=logger,
        observers=observers,
        fitness=evaluator,
    )


def main() -> None:
    """Run the fll-scheduler-ga application."""
    args = create_parser().parse_args()
    initialize_logging(args)
    config, config_parser, event_factory = setup_environment(args)
    run_preflight_checks(config, event_factory)
    ga_parameters = build_ga_parameters_from_args(args, config_parser)
    rng = setup_rng(args)
    ga = create_ga_instance(config, event_factory, ga_parameters, rng)

    if ga.run():
        summary(args, ga)
    else:
        logger.warning("Genetic algorithm did not produce a valid final schedule.")

    logger.info("fll-scheduler-ga application finished")


if __name__ == "__main__":
    main()
