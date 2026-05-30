"""Typer based CLI scheduler implementation."""

import json
import logging
import logging.config
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from fll_scheduler_ga.adapter.exporter import generate_summary
from fll_scheduler_ga.adapter.observer import LoggingObserver, RichObserver
from fll_scheduler_ga.adapter.plot import MatplotlibVisualizer
from fll_scheduler_ga.config.app_config import AppConfig, build_app_config, log_appconfig_creation_info
from fll_scheduler_ga.config.manager import ConfigManager
from fll_scheduler_ga.constants import LOGGING_CONFIG_PATH, FitnessObjective
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.genetic.ga_context import StandardGaContextFactory
from fll_scheduler_ga.genetic.stagnation import FitnessHistory, OperatorStats

if TYPE_CHECKING:
    from rich.progress import TaskID


app = typer.Typer(
    name="Tournament Scheduler",
    help="Tournament Scheduler",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()
manager = ConfigManager()
logger = logging.getLogger(__name__)


def initialize_logging(path: Path | None = None) -> None:
    """Initialize logging for the application."""
    if path is None:
        path = LOGGING_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        logging_config_dict = json.load(f)
        logging.config.dictConfig(logging_config_dict)


@app.command(name="list")
def list_configs() -> None:
    """List all available configuration files."""
    manager.refresh_list()
    active = manager.get_active_config()
    active_name = active.name if active else ""
    table = Table(title="Available Configurations")
    table.add_column("Index", justify="right", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")
    table.add_column("Status", justify="center")
    for idx, path in enumerate(manager.available):
        status = "[green]* Active[/green]" if path.name == active_name else ""
        table.add_row(str(idx), path.name, status)
    console.print(table)


@app.command(name="set")
def set_active_config(identifier: str) -> None:
    """Set the active configuration file.

    Arguments:
        identifier: The index number OR filename of the config.

    """
    selected = manager.get_config(identifier)
    manager.set_active_config(identifier)
    list_configs()
    console.print(f"[green]Successfully set: {selected.name}[/green]")


@app.command(name="add")
def add_config(
    src: Annotated[str, typer.Option("--src", "-s")], name: Annotated[str, typer.Option("--name", "-n")] = ""
) -> None:
    """Add a new configuration file.

    Arguments:
        src: Path to the source config file.
        name: Optional new name for the config file.

    """
    manager.add_config(src, name)
    list_configs()


@app.command()
def run() -> None:
    """Run a single instance of the Genetic Algorithm."""
    active_config_path = manager.get_active_config()
    console.print(f"[cyan]Initializing GA with {active_config_path.name}...[/cyan]")
    initialize_logging()
    logger.debug("Start: Tournament Scheduler.")
    start_time = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)
        try:
            ga = run_ga_engine(active_config_path, progress=progress, task_id=task)
            duration = time.perf_counter() - start_time
            best_fitness = sum(ga.fitness_history.get_last_gen_fitness())
            pareto_size = len(ga.pareto_front())
            console.print("[bold green]Run Complete![/bold green]")
            console.print(f"Duration: {duration:.2f}s")
            console.print(f"Best Fitness: {best_fitness:.4f}")
            console.print(f"Pareto Front Size: {pareto_size}")
        except (OSError, ValueError, RuntimeError) as e:
            console.print(f"[red]Run failed: {e}[/red]")
            console.print_exception()


@app.command()
def batch(count: int = typer.Argument(..., min=1, help="Number of times to run the GA")) -> None:
    """Run the GA multiple times to gather statistics.

    Arguments:
        count: Number of times to run the GA (must be >= 1).

    """
    path = manager.get_active_config()
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        batch_task = progress.add_task("[bold]Batch Progress[/bold]", total=count)
        for i in range(count):
            start_t = time.perf_counter()
            # Silence standard logs during batch
            logging.getLogger().setLevel(logging.CRITICAL)
            run_task = progress.add_task(f"Run {i + 1}/{count}", total=None)
            try:
                ga = run_ga_engine(path, progress=progress, task_id=run_task)
                duration = time.perf_counter() - start_t
                best_fitness = sum(ga.fitness_history.get_last_gen_fitness())
                pareto_size = len(ga.pareto_front())
                results.append({"run": i + 1, "fitness": best_fitness, "pareto": pareto_size, "time": duration})
            except (OSError, ValueError, RuntimeError) as e:
                console.print(f"[red]Run {i + 1} failed: {e}[/red]")
                console.print_exception()
            finally:
                logging.getLogger().setLevel(logging.INFO)
                progress.remove_task(run_task)
                progress.advance(batch_task)
    # Results Table
    table = Table(title=f"Batch Results ({path.name})")
    table.add_column("Run #", justify="right")
    table.add_column("Best Fitness", justify="right", style="green")
    table.add_column("Pareto Size", justify="right")
    table.add_column("Time (s)", justify="right")
    avg_fit = 0
    for r in results:
        table.add_row(str(r["run"]), f"{r['fitness']:.4f}", str(r["pareto"]), f"{r['time']:.2f}")
        avg_fit += r["fitness"]
    if results:
        avg_fit /= len(results)
        table.add_row("AVG", f"[bold]{avg_fit:.4f}[/bold]", "-", "-")
    console.print(table)


def run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: TaskID | None = None) -> GA:
    """Core logic to build and run the GA."""
    app_config: AppConfig = build_app_config(config_path)
    log_appconfig_creation_info(app_config)
    context_factory = StandardGaContextFactory()
    context = context_factory.build(app_config)
    trackers = ("success", "total")
    crossover_counters = {str(c): 0 for c in context.crossovers}
    mutation_counters = {str(m): 0 for m in context.mutations}
    operator_stats = OperatorStats(
        offspring=Counter(),
        crossover={tr: Counter(crossover_counters) for tr in trackers},
        mutation={tr: Counter(mutation_counters) for tr in trackers},
    )
    n_gen = app_config.genetic.parameters.generations
    n_obj = context.evaluator.n_objectives
    generation = 0
    fitness_history = FitnessHistory(
        generation=generation,
        current=np.zeros((1, n_obj), dtype=float),
        history=np.full((n_gen, n_obj), fill_value=-1, dtype=float),
    )
    generations_array = np.arange(1, n_gen + 1)
    migrate_generations = np.zeros(n_gen + 1, dtype=int)
    n_islands = app_config.genetic.parameters.num_islands
    migration_size = app_config.genetic.parameters.migration_size
    migration_interval = app_config.genetic.parameters.migration_interval
    if n_islands > 1 and migration_size > 0:
        migrate_generations[::migration_interval] = 1
    _exports = app_config.io.exports
    ga = GA(
        context=context,
        genetic_model=app_config.genetic,
        rng=app_config.rng,
        observers=(LoggingObserver(),),
        seed_file=Path(app_config.runtime.seed_file),
        save_front_only=_exports.front_only,
        generation=generation,
        operator_stats=operator_stats,
        fitness_history=fitness_history,
        generations_array=generations_array,
        migrate_generations=migrate_generations,
    )
    if progress and task_id is not None:
        ga.observers = (*tuple(ga.observers), RichObserver(progress, task_id))
    ga.run()
    output_dir = Path(_exports.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot = MatplotlibVisualizer(
        total_population=ga.total_population,
        fitness_history=ga.fitness_history.history,
        save_dir=output_dir,
        objectives=tuple(FitnessObjective),
        ref_points=ga.context.nsga3.refs.points,
        export_model=_exports,
    )
    generate_summary(ga=ga, output_dir=output_dir, export_model=_exports, plot=plot)
    return ga
