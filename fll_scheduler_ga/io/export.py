"""Module for exporting schedules to different formats."""

import argparse
import csv
import logging
from pathlib import Path

from ..data_model.location import Location
from ..data_model.team import Individual, Team
from ..data_model.time import TimeSlot
from ..genetic.fitness import FitnessEvaluator
from ..genetic.ga import GA
from ..genetic.schedule import Schedule
from ..visualize.plot import Plot
from .base_exporter import Exporter

logger = logging.getLogger(__name__)


def generate_summary(args: argparse.Namespace, ga: GA) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if not args.no_plotting:
        plot = Plot(ga)

        fitness_plot_path = output_dir / "fitness_vs_generation.png"
        plot.plot_fitness(save_dir=fitness_plot_path)

        pareto_plot_path = output_dir / "pareto_front.png"
        plot.plot_pareto_front(save_dir=pareto_plot_path)

    front = ga.pareto_front()
    front.sort(key=lambda s: (s.rank, -s.crowding))

    for i, schedule in enumerate(front, start=1):
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
        txt_output_path = txt_subdir / f"{name}_summary.txt"
        generate_summary_report(schedule, ga.evaluator, txt_output_path)

    pareto_summary_path = output_dir / "pareto_summary.csv"
    generate_pareto_summary(ga.population, ga.evaluator, pareto_summary_path)


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


def get_exporter(path: Path) -> Exporter:
    """Get the appropriate exporter based on the file extension."""
    exporter = {
        ".csv": CsvExporter,
        ".html": HtmlExporter,
    }.get(path.suffix.lower(), None)
    if exporter is None:
        logger.warning("No exporter found for file extension %s. Defaulting to CSV.", path.suffix)
        return CsvExporter()
    return exporter()


class GridBasedExporter(Exporter):
    """Base class for exporters that render a grid-based schedule."""

    def _build_grid_data(
        self, schedule: Individual
    ) -> tuple[list[TimeSlot], list[Location], dict[tuple[TimeSlot, Location], Team]]:
        """Build the common grid data structure from a schedule."""
        grid_lookup = {(e.timeslot, e.location): team for e, team in schedule.items()}
        timeslots = sorted({i[0] for i in grid_lookup}, key=lambda ts: ts.start)
        locations = sorted(
            {i[1] for i in grid_lookup}, key=lambda loc: (loc.identity, loc.side if hasattr(loc, "side") else 0)
        )
        return timeslots, locations, grid_lookup


class CsvExporter(GridBasedExporter):
    """Exporter for schedules in CSV format."""

    def export(self, schedule: Individual, filename: str) -> None:
        """Export the given schedule to a CSV file.

        Args:
            schedule (Individual): A mapping of Event to Team, representing the schedule.
            filename (str): The name of the file to write the schedule to.

        """
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return

        filename = Path(filename) if isinstance(filename, str) else filename

        schedule_type = self._group_by_type(schedule)

        csv_parts = []
        for rt, values in schedule_type.items():
            csv_parts.extend(self.render_grid(rt, values))

        try:
            with filename.open("w", newline="", encoding="utf-8") as csvfile:
                csv.writer(csvfile).writerows(csv_parts)
            logger.debug("Schedule successfully exported to %s", filename)
        except OSError:
            logger.exception("Failed to write schedule to file %s", filename)

    def render_grid(self, title: str, schedule: Individual) -> list[str]:
        """Write a single schedule grid to a CSV writer."""
        rows = []
        rows.append([title])
        if not schedule:
            return [*rows, "No events scheduled for this round type.", []]

        timeslots, locations, grid_lookup = self._build_grid_data(schedule)

        header = ["Time"] + [str(loc) for loc in locations]
        rows.append(header)

        for time_slot in timeslots:
            r = [time_slot.start_str]
            for location in locations:
                team_or_match = grid_lookup.get((time_slot, location))
                if isinstance(team_or_match, Team):
                    r.append(str(team_or_match.identity))
                else:
                    r.append("")
            rows.append(r)
        rows.append([])
        return rows


class HtmlExporter(GridBasedExporter):
    """Exporter for schedules in HTML format."""

    def export(self, schedule: Individual, filename: str) -> None:
        """Export the schedule to a self-contained HTML file with CSS."""
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return

        filename = Path(filename) if isinstance(filename, str) else filename

        html_parts = [
            """
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
        ]

        schedule_by_type = self._group_by_type(schedule)

        for rt, values in schedule_by_type.items():
            html_parts.extend(self.render_grid(rt, values))

        html_parts.append("</body></html>")

        try:
            with filename.open("w", encoding="utf-8") as f:
                f.write("".join(html_parts))
            logger.debug("Schedule successfully exported to %s", filename)
        except OSError:
            logger.exception("Failed to write schedule to file")

    def render_grid(self, title: str, schedule: Individual) -> list[str]:
        """Render a single schedule grid as an HTML table."""
        if not schedule:
            return [f"<h2>{title}</h2><p>No events scheduled.</p>"]

        timeslots, locations, grid_lookup = self._build_grid_data(schedule)

        html = [f"<h2>{title}</h2>", "<table>", "<thead>", "<tr><th>Time</th>"]
        for location in locations:
            html.extend(f"<th>{location!s}</th>")
        html.extend(["</tr>", "</thead>", "<tbody>"])

        for time_slot in timeslots:
            html.append(f"<tr><td>{time_slot.start_str}</td>")
            for location in locations:
                team_or_match = grid_lookup.get((time_slot, location))
                if isinstance(team_or_match, Team):
                    html.append(f"<td>{team_or_match.identity}</td>")
                else:
                    html.append("<td></td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return html
