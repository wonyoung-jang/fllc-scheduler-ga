"""Main entry point for the fll-scheduler-ga package."""

import argparse
import logging
from pathlib import Path
from random import Random

from fll_scheduler_ga.config.config import load_tournament_config
from fll_scheduler_ga.data_model.event import EventFactory, EventMap
from fll_scheduler_ga.data_model.team import Team, TeamFactory
from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
from fll_scheduler_ga.genetic.ga import GA
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
        "config_file": "fll_scheduler_ga/config/config.ini",
        "output_file": "outputs/schedule",
        "log_file": "fll_scheduler_ga.log",
        "file_log_level": "DEBUG",
        "console_log_level": "INFO",
        "population_size": 16,
        "generations": 128,
        "elite_size": 3,
        "selection_size": 4,
        "crossover_chance": 0.5,
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
        "--output_file",
        type=str,
        default=_default_values["output_file"],
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=_default_values["log_file"],
        help="Path to the log file.",
    )
    parser.add_argument(
        "--file_log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["file_log_level"],
        help="Logging level.",
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=_default_values["console_log_level"],
        help="Logging level for the console output.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=_default_values["population_size"],
        help="Population size for the GA.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=_default_values["generations"],
        help="Number of generations to run.",
    )
    parser.add_argument(
        "--elite_size",
        type=int,
        default=_default_values["elite_size"],
        help="Number of elite individuals.",
    )
    parser.add_argument(
        "--selection_size",
        type=int,
        default=_default_values["selection_size"],
        help="Size of parent selection.",
    )
    parser.add_argument(
        "--crossover_chance",
        type=float,
        default=_default_values["crossover_chance"],
        help="Chance of crossover (0.0 to 1.0).",
    )

    return parser


def build_ga_parameters_from_args(args: argparse.Namespace) -> GaParameters:
    """Build a GaParameters, overriding defaults with any provided CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        GaParameters: Parameters for the genetic algorithm.

    """
    cli_args = {
        "population_size": args.population_size,
        "generations": args.generations,
        "elite_size": args.elite_size,
        "selection_size": args.selection_size,
        "crossover_chance": args.crossover_chance,
    }
    provided_args = {k: v for k, v in cli_args.items() if v is not None}
    return GaParameters(**provided_args)


def main() -> None:
    """Run the fll-scheduler-ga application."""
    parser = create_parser()
    args = parser.parse_args()

    initialize_logging(args)

    try:
        config = load_tournament_config(args.config_file)
        logger.info("Loaded tournament configuration: %s", config)
    except (FileNotFoundError, KeyError):
        logger.exception("Error loading configuration")
        return

    try:
        run_preflight_checks(config)
    except ValueError:
        logger.exception("Preflight checks failed")
        return

    rng_seed = 123456789
    rng = Random(rng_seed)

    event_factory = EventFactory(config)
    event_conflicts = EventMap(event_factory)
    team_factory = TeamFactory(config, event_conflicts.conflicts)
    ga_parameters = build_ga_parameters_from_args(args)
    selection = TournamentSelectionNSGA2(ga_parameters, rng)
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

    evaluator = FitnessEvaluator(
        config=config,
    )

    observers = [
        LoggingObserver(logger),
        TqdmObserver(logger),
    ]

    ga = GA(
        ga_parameters=ga_parameters,
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
        fitness=evaluator,
    )

    if pareto_front := ga.run():
        for schedule in pareto_front:
            logger.debug(
                "Schedule %d: Rank = %d, Fitness = %s, Crowding Distance = %.4f",
                id(schedule),
                schedule.rank,
                schedule.fitness,
                schedule.crowding_distance,
            )

        primary_output_path = Path(args.output_file)

        output_dir = primary_output_path.parent

        plot = Plot(ga)
        plot.plot_fitness(save_dir=output_dir / "fitness_vs_generation.png")
        plot.plot_pareto_front(save_dir=output_dir / "pareto_front_tradeoffs.png")

        schedules_to_export = {
            "best_break_time": max(pareto_front, key=lambda s: s.fitness[0]),
            "best_opponent_variety": max(pareto_front, key=lambda s: s.fitness[1]),
            "best_table_consistency": max(pareto_front, key=lambda s: s.fitness[2]),
            "most_balanced": max(pareto_front, key=lambda s: (s.crowding_distance, sum(s.fitness))),
            "best_break_time_and_opponent_variety": max(pareto_front, key=lambda s: (s.fitness[0], s.fitness[1])),
            "best_break_time_and_table_consistency": max(pareto_front, key=lambda s: (s.fitness[0], s.fitness[2])),
            "best_opponent_variety_and_table_consistency": max(
                pareto_front, key=lambda s: (s.fitness[1], s.fitness[2])
            ),
            "best_opponent_variety_and_break_time": max(pareto_front, key=lambda s: (s.fitness[1], s.fitness[0])),
            "best_table_consistency_and_break_time": max(pareto_front, key=lambda s: (s.fitness[2], s.fitness[0])),
            "best_table_consistency_and_opponent_variety": max(
                pareto_front, key=lambda s: (s.fitness[2], s.fitness[1])
            ),
        }

        for name, schedule in schedules_to_export.items():
            output_path = primary_output_path.with_name(
                f"{primary_output_path.stem}_{name}{primary_output_path.suffix}"
            )

            csv_path = output_path.with_suffix(".csv")
            csv_exporter = get_exporter(csv_path)
            csv_exporter.export(schedule, csv_path)

            html_path = output_path.with_suffix(".html")
            html_exporter = get_exporter(html_path)
            html_exporter.export(schedule, html_path)

            report_path = output_path.with_suffix(".txt")
            generate_summary_report(schedule, evaluator, report_path)

        generate_pareto_summary(pareto_front, evaluator, output_dir / "pareto_summary.txt")
    else:
        logger.warning("Genetic algorithm did not produce a valid final schedule.")

    logger.info("fll-scheduler-ga application finished")


def initialize_logging(args: argparse.Namespace) -> None:
    """Initialize logging for the application."""
    file_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(args.file_log_level)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(name)s] %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.console_log_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s[%(name)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    args_str = "\n".join(f"\t{k} = {v}" for k, v in args.__dict__.items())
    logger.info("Starting fll-scheduler-ga application with args:\n%s", args_str)


def generate_summary_report(schedule: Schedule, evaluator: FitnessEvaluator, path: Path) -> None:
    """Generate a text summary report for a single schedule."""
    obj_names = list(evaluator.soft_constraints)
    scores = schedule.fitness
    with path.open("w", encoding="utf-8") as f:
        f.write(f"--- FLL Scheduler GA Summary Report ({id(schedule)}) ---\n\n")
        f.write("Objective Scores:\n")
        for name, score in zip(obj_names, scores, strict=False):
            f.write(f"  - {name}: {score:.4f}\n")

        f.write("\nNotes:\n")
        all_teams: list[Team] = list(schedule.all_teams)
        worst_team = min(all_teams, key=lambda t: t.score_break_time())
        f.write(f"  - Team with worst break time distribution: Team {worst_team.identity}\n")
        worst_team = min(all_teams, key=lambda t: t.score_break_time())


def generate_pareto_summary(pareto_front: list[Schedule], evaluator: FitnessEvaluator, path: Path) -> None:
    """Generate a summary of the Pareto front."""
    obj_names = list(evaluator.soft_constraints)
    pareto_front.sort(key=lambda s: (s.crowding_distance, s.fitness[0], s.fitness[1], s.fitness[2]), reverse=True)
    with path.open("w", encoding="utf-8") as f:
        for i, schedule in enumerate(pareto_front, start=1):
            f.write(f"Schedule {i}: ({id(schedule)}) - Crowding Distance: {schedule.crowding_distance:.4f} ")
            for name, score in zip(obj_names, schedule.fitness, strict=False):
                f.write(f"| {name}: {score:.4f} |")
            f.write(f"| Sum: {sum(schedule.fitness):.4f}|\n")


if __name__ == "__main__":
    main()
