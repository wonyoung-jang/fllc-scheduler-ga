"""CLI argument parsing."""

from argparse import ArgumentParser
from importlib.metadata import version


def create_parser() -> ArgumentParser:
    """Create the argument parser for the application.

    Returns:
        ArgumentParser: Configured argument parser.

    """
    _default_values = {
        "cmap_name": "viridis",
        "config_file": "fll_scheduler_ga/config.ini",
        "front_only": True,
        "log_file": "fll_scheduler_ga.log",
        "loglevel_console": "INFO",
        "loglevel_file": "DEBUG",
        "no_plotting": False,
        "output_dir": "fllc_schedule_outputs",
        "seed_file": "fllc_genetic.pkl",
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
    output_group.add_argument(
        "--cmap_name",
        type=str,
        default=_default_values["cmap_name"],
        help="Name of the colormap to use for plotting.",
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
