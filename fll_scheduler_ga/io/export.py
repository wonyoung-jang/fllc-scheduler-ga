"""Module for exporting schedules to different formats."""

import argparse
import csv
import logging
from pathlib import Path

from ..genetic.fitness import FitnessObjective
from ..genetic.ga import GA
from ..genetic.schedule import Schedule
from ..visualize.plot import Plot
from .base_exporter import Exporter, GridBasedExporter

logger = logging.getLogger(__name__)


def generate_summary(args: argparse.Namespace, ga: GA) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if not args.no_plotting:
        plot = Plot(ga)

        fitness_plot_path = output_dir / "fitness_vs_generation.png"
        plot.plot_fitness(
            title="Average Fitness over Generations",
            xlabel="Generation",
            ylabel="Average Fitnesses",
            save_dir=fitness_plot_path,
        )

        pareto_plot_path = output_dir / "pareto_front.png"
        plot.plot_pareto_front(
            title="Pareto Front: Trade-offs",
            save_dir=pareto_plot_path,
        )

    front = sorted(ga.pareto_front(), key=lambda s: (s.rank, -sum(s.fitness)))

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
        generate_summary_report(schedule, ga.context.evaluator.objectives, txt_output_path)

    pareto_summary_path = output_dir / "pareto_summary.csv"
    generate_pareto_summary(ga.total_population, ga.context.evaluator.objectives, pareto_summary_path)


def generate_summary_report(schedule: Schedule, objectives: list[FitnessObjective], path: Path) -> None:
    """Generate a text summary report for a single schedule."""
    len_objectives = [len(name) for name in objectives]
    max_len_obj = max(len_objectives, default=0) + 1

    try:
        with path.open("w", encoding="utf-8") as f:
            f.write(f"--- FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)}) ---\n\n")
            f.write("Objective Scores:\n")

            for name, score in zip(objectives, schedule.fitness, strict=False):
                f.write(f"  - {name:<{max_len_obj}}: {score:.6f}\n")

            f.write("\n")
            f.write(f"{'Total':<{max_len_obj}}: {sum(schedule.fitness):.6f}\n")
            f.write(f"{'Average':<{max_len_obj}}: {sum(schedule.fitness) / len(schedule.fitness):.6f}\n\n")

            all_teams = schedule.all_teams()
            total_fitnesses = [sum(t.fitness) for t in all_teams]
            max_team_f = max(total_fitnesses)
            min_team_f = min(total_fitnesses)

            f.write("Team fitnesses (sorted by total fitness descending):\n\n")

            f.write(f"Max     : {max_team_f:.6f}\n")
            f.write(f"Min     : {min_team_f:.6f}\n")
            f.write(f"Range   : {max_team_f - min_team_f:.6f}\n")
            f.write(f"Average : {sum(total_fitnesses) / len(total_fitnesses):.6f}\n\n")

            normalized_teams = schedule.normalize_teams()

            objectives_header = (f"{name:<{len_objectives[i] + 1}}" for i, name in enumerate(objectives))
            objectives_header_str = "|".join(objectives_header)
            header = f"{'Team':<5}|{objectives_header_str}|Sum\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

            for t in sorted(all_teams, key=lambda x: -sum(x.fitness)):
                fitness_row = (f"{score:<{len_objectives[i] + 1}.4f}" for i, score in enumerate(t.fitness))
                fitness_str = "|".join(fitness_row)
                team_id = normalized_teams.get(t.identity)
                f.write(f"{team_id:<5}|{fitness_str}|{sum(t.fitness):.4f}\n")
    except OSError:
        logger.exception("Failed to write summary report to file %s", path)


def generate_pareto_summary(front: list[Schedule], objectives: list[FitnessObjective], path: Path) -> None:
    """Generate a summary of the Pareto front."""
    schedule_enum_digits = len(str(len(front)))
    front.sort(key=lambda s: (s.rank, -sum(s.fitness)))
    try:
        with path.open("w", encoding="utf-8") as f:
            f.write("Schedule, ID, Hash, Rank, ")

            for name in objectives:
                f.write(f"{name}, ")

            f.write("Sum\n")
            for i, schedule in enumerate(front, start=1):
                f.write(f"{i:0{schedule_enum_digits}}, {id(schedule)}, {hash(schedule)}, {schedule.rank}, ")

                for score in schedule.fitness:
                    f.write(f"{score:.4f}, ")

                f.write(f"{sum(schedule.fitness):.4f}\n")
    except OSError:
        logger.exception("Failed to write Pareto summary to file %s", path)


def get_exporter(path: Path) -> Exporter:
    """Get the appropriate exporter based on the file extension."""
    exporter_map = {
        ".csv": CsvExporter,
        ".html": HtmlExporter,
    }

    exporter = exporter_map.get(path.suffix.lower(), None)

    if exporter is None:
        logger.warning("No exporter found for file extension %s. Defaulting to CSV.", path.suffix)
        return CsvExporter()
    return exporter()


class CsvExporter(GridBasedExporter):
    """Exporter for schedules in CSV format."""

    def export(self, schedule: Schedule, filename: str) -> None:
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

    def render_grid(self, title: str, schedule: Schedule) -> list[str]:
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
                team = grid_lookup.get((time_slot, location))
                if isinstance(team, int):
                    r.append(str(team))
                else:
                    r.append("")
            rows.append(r)
        rows.append([])
        return rows


class HtmlExporter(GridBasedExporter):
    """Exporter for schedules in HTML format."""

    def export(self, schedule: Schedule, filename: str) -> None:
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

    def render_grid(self, title: str, schedule: Schedule) -> list[str]:
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
                team = grid_lookup.get((time_slot, location))
                if isinstance(team, int):
                    html.append(f"<td>{team}</td>")
                else:
                    html.append("<td></td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return html
