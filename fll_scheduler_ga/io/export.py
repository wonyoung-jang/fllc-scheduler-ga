"""Module for exporting schedules to different formats."""

from __future__ import annotations

from csv import writer
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

from ..config.constants import FitnessObjective
from ..visualize.plot import Plot
from .base_exporter import GridBasedExporter

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterator

    from ..data_model.config import RoundType
    from ..data_model.schedule import Individual, Population, Schedule
    from ..genetic.ga import GA

logger = getLogger(__name__)


def generate_summary(args: Namespace, ga: GA) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    output_dir = Path(args.output_dir)
    subdirs = setup_output_dirs(output_dir)
    generate_plots(ga, output_dir, args)
    schedules = ga.pareto_front() if args.front_only else ga.total_population
    schedules.sort(key=lambda s: (s.rank, -sum(s.fitness)))
    export_all_schedules(schedules, subdirs, ga)
    generate_pareto_summary(ga.total_population, output_dir / "pareto_summary.csv")


def setup_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Set up the output directories for the different export formats."""
    if output_dir.exists():
        logger.debug("Output directory %s already exists. Clearing contents.", output_dir)
        rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory: %s", output_dir)

    subdir_names = ("csv", "html", "txt", "team_schedules")
    subdirs = {name: output_dir / name for name in subdir_names}
    for sd in subdirs.values():
        sd.mkdir(parents=True, exist_ok=True)
    return subdirs


def export_all_schedules(schedules: list[Schedule], subdirs: dict[str, Path], ga: GA) -> None:
    """Export all schedules to the different formats."""
    time_fmt = ga.context.app_config.tournament.time_fmt
    event_factory = ga.context.event_factory
    csv_exporter = CsvExporter(time_fmt, event_factory)
    html_exporter = HtmlExporter(time_fmt, event_factory)

    for i, sched in enumerate(schedules, start=1):
        name = f"front_{sched.rank}_schedule_{i}"

        csv_exporter.export(sched, subdirs["csv"] / f"{name}.csv")
        html_exporter.export(sched, subdirs["html"] / f"{name}.html")
        generate_summary_report(sched, subdirs["txt"] / f"{name}_summary.txt")
        generate_team_schedules(sched, subdirs["team_schedules"] / f"{name}_team_schedule.csv", ga)


def generate_plots(ga: GA, output_dir: Path, args: Namespace) -> None:
    """Generate plots for the GA results."""
    if not args.no_plotting and ga.total_population:
        Plot(ga, output_dir, args.cmap_name).plot()


def generate_summary_report(schedule: Schedule, path: Path) -> None:
    """Generate a text summary report for a single schedule."""
    try:
        with path.open("w", encoding="utf-8") as f:
            objectives = list(FitnessObjective)
            len_objectives = [len(name) for name in objectives]
            max_len_obj = max(len_objectives, default=0) + 1
            f.write(f"FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)})\n")

            f.write("\n")
            f.write("Schedule attributes:\n")
            f.write("--------------------\n")
            slots = (
                s for s in schedule.__slots__ if not s.startswith("_") and s not in ("fitness", "teams", "schedule")
            )
            for slot in slots:
                f.write(f"{slot}: {getattr(schedule, slot)}\n")

            f.write("\n")
            f.write("Schedule objective scores:\n")
            f.write("--------------------------\n")
            for name, score in zip(objectives, schedule.fitness, strict=True):
                f.write(f"{name:<{max_len_obj}}: {score:.6f}\n")
            f.write(f"{'-' * (max_len_obj + 15)}\n")
            f.write(f"{'Total':<{max_len_obj}}: {sum(schedule.fitness):.6f}\n")
            f.write(f"{'Percentage':<{max_len_obj}}: {sum(schedule.fitness) / len(schedule.fitness):.2%}\n")

            all_teams = schedule.teams
            team_fits = schedule.team_fitnesses
            total_fitnesses = team_fits.sum(axis=1)
            max_team_f = max(total_fitnesses)
            min_team_f = min(total_fitnesses)

            f.write("\n")
            f.write("Team fitnesses (sorted by total fitness descending):\n")
            f.write("----------------------------------------------------\n")
            f.write(f"Max     : {max_team_f:.6f}\n")
            f.write(f"Min     : {min_team_f:.6f}\n")
            f.write(f"Range   : {max_team_f - min_team_f:.6f}\n")
            f.write(f"Average : {sum(total_fitnesses) / len(total_fitnesses):.6f}\n")

            f.write("\n")
            objectives_header = (f"{name:<{len_objectives[i] + 1}}" for i, name in enumerate(objectives))
            objectives_header_str = "|".join(objectives_header)
            header = f"{'Team':<5}|{objectives_header_str}|Sum\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            normalized_teams = schedule.normalized_teams()
            # use the numpy array
            for t, fit in sorted(zip(all_teams, team_fits, strict=True), key=lambda x: -x[1].sum()):
                fitness_row = (f"{score:<{len_objectives[i] + 1}.4f}" for i, score in enumerate(fit))
                fitness_str = "|".join(fitness_row)
                team_id = normalized_teams.get(t.idx)
                if team_id is None:
                    team_id = -1
                f.write(f"{team_id:<5}|{fitness_str}|{sum(fit):.4f}\n")
    except OSError:
        logger.exception("Failed to write summary report to file %s", path)


def generate_team_schedules(schedule: Schedule, path: Path, ga: GA) -> None:
    """Generate a CSV file with team schedules, sorted by team IDs."""
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            event_factory = ga.context.event_factory
            config = ga.context.app_config.tournament
            event_map = event_factory.as_mapping()
            rows = []
            headers = ["Team"]

            for r in sorted(config.rounds, key=lambda r: r.start_time):
                rt = r.roundtype
                count = config.round_requirements.get(rt, 0)
                if count == 1:
                    headers.append(f"{rt.capitalize()}")
                    headers.append("")
                else:
                    for i in range(1, count + 1):
                        headers.append(f"{rt.capitalize()} {i}")
                        headers.append("")

            rows.append(headers)

            for team_id, events in sorted(schedule.team_events.items()):
                row = [team_id + 1]
                for event_id in sorted(events):
                    event = event_map.get(event_id)
                    row.append(str(event.timeslot))
                    row.append(str(event.location))
                rows.append(row)

            writer(f).writerows(rows)
    except OSError:
        logger.exception("Failed to write team schedules to file %s", path)


def generate_pareto_summary(pop: Population, path: Path) -> None:
    """Generate a summary of the Pareto front."""
    try:
        with path.open("w", encoding="utf-8") as f:
            schedule_enum_digits = len(str(len(pop)))
            f.write("Schedule,ID,Hash,Rank,")
            for name in list(FitnessObjective):
                f.write(f"{name},")

            f.write("Sum,Origin,Mutations,Clones")
            f.write("\n")
            for i, schedule in enumerate(sorted(pop, key=lambda s: (s.rank, -sum(s.fitness))), start=1):
                f.write(f"{i:0{schedule_enum_digits}},{id(schedule)},{hash(schedule)},{schedule.rank},")

                for score in schedule.fitness:
                    f.write(f"{score:.4f},")

                f.write(f"{sum(schedule.fitness):.4f},")
                f.write(f"{schedule.origin if schedule.origin is not None else ''},")
                f.write(f"{schedule.mutations if schedule.mutations is not None else ''},")
                f.write(f"{schedule.clones if schedule.clones is not None else ''}")
                f.write("\n")
    except OSError:
        logger.exception("Failed to write Pareto summary to file %s", path)


@dataclass(slots=True)
class CsvExporter(GridBasedExporter):
    """Exporter for schedules in CSV format."""

    def render_grid(self, schedule_by_type: dict[RoundType, Individual]) -> Iterator[list[str]]:
        """Write a single schedule grid to a CSV writer."""
        for title, schedule in schedule_by_type.items():
            yield [title]
            if not schedule:
                yield ["No events scheduled for this round type.", []]
                continue
            timeslots, locations, grid_lookup = self._build_grid_data(schedule)
            yield ["Time"] + [str(loc) for loc in locations]
            for ts in timeslots:
                r = [ts.start.strftime(self.time_fmt)]
                for loc in locations:
                    team = grid_lookup.get((ts, loc))
                    if isinstance(team, int):
                        r.append(str(team))
                    else:
                        r.append("")
                yield r
            yield []

    def write_to_file(self, schedule_by_type: dict[RoundType, Individual], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", newline="", encoding="utf-8") as csvfile:
            writer(csvfile).writerows(self.render_grid(schedule_by_type))


@dataclass(slots=True)
class HtmlExporter(GridBasedExporter):
    """Exporter for schedules in HTML format."""

    def render_grid(self, schedule_by_type: dict[RoundType, Individual]) -> Iterator[str]:
        """Render a single schedule grid as an HTML table."""
        yield """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Tournament Schedule</title>
                <style>
                    body {
                        font-family: Roboto, Helvetica, Arial, sans-serif;
                        line-height: 1.6; color: #333;
                    }
                    table {
                        border-collapse: collapse;
                        margin-bottom: 2em;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    th, td {
                        border: 1px solid #ccc;
                        padding: 8px 12px;
                        text-align: center;
                    }
                    th {
                        background-color: #f2f2f2;
                        font-weight: 600;
                    }
                    h1, h2 {
                        color: #1a1a1a;
                        border-bottom: 2px solid #eee;
                        padding-bottom: 0.3em;
                    }
                    .container {
                        max-width: 95%;
                        margin: auto;
                        padding: 2em;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                <h1>Tournament Schedule</h1>
        """
        for title, schedule in schedule_by_type.items():
            yield f"<h2>{title}</h2>"
            if not schedule:
                yield "<p>No events scheduled.</p>"
                continue

            timeslots, locations, grid_lookup = self._build_grid_data(schedule)
            yield "<table><thead><tr><th>Time</th>"
            for loc in locations:
                yield f"<th>{loc!s}</th>"
            yield "</tr></thead><tbody>"

            for ts in timeslots:
                yield f"<tr><td>{ts.start.strftime(self.time_fmt)}</td>"
                for loc in locations:
                    team = grid_lookup.get((ts, loc))
                    if isinstance(team, int):
                        yield f"<td>{team}</td>"
                    else:
                        yield "<td></td>"
                yield "</tr>"
            yield "</tbody></table>"

    def write_to_file(self, schedule_by_type: dict[RoundType, Individual], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", encoding="utf-8") as f:
            f.write("".join(self.render_grid(schedule_by_type)))
