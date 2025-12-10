"""Main cli api for the fll-scheduler-ga package."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from fll_scheduler_ga.config._config_manager import ConfigManager
from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.engine import init_logging
from fll_scheduler_ga.genetic.ga import GA
from fll_scheduler_ga.genetic.ga_context import GaContext
from fll_scheduler_ga.io.observers import RichObserver

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Genetic Algorithm Scheduler", no_args_is_help=True)
console = Console()
manager = ConfigManager()


def _get_active_config() -> Path:
    """Retrieve active config or raise Exit."""
    path = manager.get_active_config()
    if not path:
        console.print("[red]No active configuration selected. Use 'load' first.[/red]")
        raise typer.Exit(code=1)
    return path


def _run_ga_engine(config_path: Path, progress: Progress | None = None, task_id: int | None = None) -> GA:
    """Core logic to build and run the GA."""
    app_config = AppConfig.build(config_path)
    context = GaContext.build(app_config)
    ga = GA.build(context)

    if progress and task_id is not None:
        existing = list(ga.observers)
        existing.append(RichObserver(progress, task_id))
        ga.observers = tuple(existing)

    ga.run()
    return ga


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
def add_config(src: str, name: str | None = None) -> None:
    """Add a new configuration file.

    Arguments:
        src: Path to the source config file.
        name: Optional new name for the config file.

    """
    manager.add_config(src, name)
    list_configs()


@app.command()
def show() -> None:
    """Show details of the currently active configuration."""
    path = _get_active_config()
    try:
        app_config = AppConfig.build(path)
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")

        p = app_config.genetic.parameters
        t = app_config.tournament

        grid.add_row(f"[bold]Generations:[/bold] {p.generations}", f"[bold]Pop Size:[/bold] {p.population_size}")
        grid.add_row(f"[bold]Islands:[/bold] {p.num_islands}", f"[bold]Mutation:[/bold] {p.mutation_chance}")
        grid.add_row(f"[bold]Teams:[/bold] {t.num_teams}", f"[bold]Rounds:[/bold] {len(t.rounds)}")

        console.print(Panel(grid, title=f"Config: {path.name}", border_style="blue"))
    except (OSError, ValueError, RuntimeError):
        console.print_exception()


@app.command()
def run() -> None:
    """Run a single instance of the Genetic Algorithm."""
    active_config_path = manager.get_active_config()
    console.print(f"[cyan]Initializing GA with {active_config_path.name}...[/cyan]")

    app_config = AppConfig.build(active_config_path)
    init_logging(app_config)

    start_time = time.time()

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
            ga = _run_ga_engine(active_config_path, progress=progress, task_id=task)

            duration = time.time() - start_time
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
    """Run the GA multiple times to gather statistics."""
    path = _get_active_config()
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
            start_t = time.time()
            # Silence standard logs during batch
            logging.getLogger().setLevel(logging.CRITICAL)

            run_task = progress.add_task(f"Run {i + 1}/{count}", total=None)

            try:
                ga = _run_ga_engine(path, progress=progress, task_id=run_task)

                duration = time.time() - start_t
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


if __name__ == "__main__":
    app()
