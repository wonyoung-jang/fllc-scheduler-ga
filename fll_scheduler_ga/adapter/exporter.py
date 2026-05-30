"""Module for exporting schedules to different formats."""

import asyncio
import csv
import html
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import FitnessObjective

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fll_scheduler_ga.adapter.plot import Visualizer
    from fll_scheduler_ga.config.pydantic_schemas import ExportModel
    from fll_scheduler_ga.domain.event import EventProperties
    from fll_scheduler_ga.domain.location import Location
    from fll_scheduler_ga.domain.schedule import Schedule
    from fll_scheduler_ga.domain.timeslot import TimeSlot
    from fll_scheduler_ga.genetic.ga import GA

logger = getLogger(__name__)


def normalize_teams(schedule: np.ndarray, team_ids: dict[int, str]) -> np.ndarray:
    """Normalize the schedule by reassigning team identities based on their order of appearance."""
    normalized = np.full(len(team_ids), -1, dtype=int)
    count = 1
    for team in schedule:
        if team != -1 and normalized[team] == -1:
            normalized[team] = team_ids.get(count, count)
            count += 1
    return normalized


@dataclass(slots=True)
class ScheduleExporter(ABC):
    """Abstract base class for exporting schedules."""

    time_fmt: str
    team_identities: dict[int, str]
    event_properties: EventProperties

    async def export(self, schedule: Schedule, path: Path) -> None:
        """Export the schedule to a given filename."""
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return
        schedule_by_type = self._group_by_type(schedule)
        try:
            await self.write_to_file(schedule_by_type, path)
            logger.debug("Schedule successfully exported to %s", path)
        except OSError:
            logger.exception("Failed to export schedule to %s", path)

    def _group_by_type(self, schedule: Schedule) -> dict[str, dict[int, int]]:
        """Group the schedule by round type."""
        grouped = {}
        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        for event, team in enumerate(schedule.schedule):
            if team == -1:
                continue
            rt = self.event_properties.roundtype[event]
            grouped.setdefault(rt, {})
            grouped[rt][event] = normalized_teams[team]
        return grouped

    def _build_grid_data(
        self, schedule: dict[int, int]
    ) -> tuple[list[TimeSlot], list[Location], dict[tuple[TimeSlot, Location], int]]:
        """Build the common grid data structure from a schedule."""
        grid_lookup = {}
        for event, team in schedule.items():
            ts = self.event_properties.timeslot[event]
            loc = self.event_properties.location[event]
            grid_lookup[(ts, loc)] = team
        timeslots: list[TimeSlot] = sorted(
            {i[0] for i in grid_lookup},
            key=lambda ts: ts.start,
        )
        locations: list[Location] = sorted(
            {i[1] for i in grid_lookup},
            key=lambda loc: (
                loc.name,
                loc.side if loc.side != -1 else 0,
            ),
        )
        return timeslots, locations, grid_lookup

    def get_table_data(self, schedule_dict: dict[int, int]) -> list[list[str]]:
        """Generate a 2D matrix of strings representing the grid for a single round type."""
        if not schedule_dict:
            return []
        timeslots, locations, grid_lookup = self._build_grid_data(schedule_dict)
        # Header Row
        header = ["Time"] + [str(loc) for loc in locations]
        matrix = [header]
        # Data Rows
        for ts in timeslots:
            ts_str = ts.start.strftime(self.time_fmt) if ts.start else "N/A"
            row = [ts_str]
            for loc in locations:
                team = grid_lookup.get((ts, loc))
                row.append(str(team) if team is not None else "")
            matrix.append(row)
        return matrix

    @abstractmethod
    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[str | list[str]]: ...
    @abstractmethod
    async def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None: ...


class CsvScheduleExporter(ScheduleExporter):
    """Exporter for schedules in CSV format."""

    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[list[str]]:
        """Write a single schedule grid as CSV rows."""
        if not schedule_dict:
            yield ["No events scheduled for this round type."]
            yield []
            return
        data = self.get_table_data(schedule_dict)
        yield from data
        yield []

    async def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for title, schedule_dict in schedule_by_type.items():
                writer.writerow([title])
                writer.writerows(self.render_grid(schedule_dict))


class HtmlScheduleExporter(ScheduleExporter):
    """Exporter for schedules in HTML format."""

    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[str]:
        """Render a single schedule grid as an HTML table."""
        if not schedule_dict:
            yield "<p>No events scheduled.</p>"
            return
        data = self.get_table_data(schedule_dict)
        # Table Start
        yield "<table>"
        # Thead
        yield "<thead><tr>"
        for cell in data[0]:
            yield f"<th>{html.escape(cell)}</th>"
        yield "</tr></thead>"
        # Tbody
        yield "<tbody>"
        for row in data[1:]:
            yield "<tr>"
            for cell in row:
                # First column is time, others are locations/teams
                tag = "td"
                yield f"<{tag}>{html.escape(cell)}</{tag}>"
            yield "</tr>"
        yield "</tbody>"
        # Table End
        yield "</table>"

    async def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", encoding="utf-8") as f:
            f.write(self._get_html_start())
            for title, schedule_dict in schedule_by_type.items():
                f.write(f"<h2>{html.escape(title)}</h2>")
                f.write("".join(self.render_grid(schedule_dict)))
            f.write(self._get_html_end())

    def _get_html_start(self) -> str:
        return """
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
                        width: 100%;
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

    def _get_html_end(self) -> str:
        return """
                </div>
            </body>
            </html>
        """


def generate_summary(
    ga: GA,
    output_dir: Path,
    export_model: ExportModel,
    plot: Visualizer,
) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    subdirs = OutputDirManager(output_dir).subdirs
    total_pop = ga.total_population
    if not export_model.no_plotting and total_pop:
        plot.plot()
    schedules = ga.pareto_front() if export_model.front_only else total_pop
    schedules.sort(key=lambda s: (s.rank, -sum(s.fitness)))
    time_fmt = ga.context.app_config.tournament.time_fmt
    event_properties = ga.context.event_properties
    exporters = get_exporters(export_model, subdirs, time_fmt, event_properties, ga)
    export_manager = ExportManager(schedules=schedules, exporters=exporters)
    asyncio.run(export_manager.export_all())
    if export_model.pareto_summary:
        pareto_summary_gen = ParetoSummaryGenerator()
        pareto_summary_gen.export(total_pop, output_dir / "pareto_summary.csv")


def get_exporters(
    export_model: ExportModel,
    subdirs: dict[str, Path],
    time_fmt: str,
    event_properties: EventProperties,
    ga: GA,
) -> tuple:
    """Get the list of exporters based on the export model."""
    exporters = []
    if export_model.schedules_csv:
        exporters.append(
            (
                CsvScheduleExporter(
                    time_fmt=time_fmt,
                    team_identities=export_model.team_identities,
                    event_properties=event_properties,
                ),
                subdirs["csv"],
                "csv",
            )
        )
    if export_model.schedules_html:
        exporters.append(
            (
                HtmlScheduleExporter(
                    time_fmt=time_fmt,
                    team_identities=export_model.team_identities,
                    event_properties=event_properties,
                ),
                subdirs["html"],
                "html",
            )
        )
    if export_model.summary_reports:
        exporters.append(
            (
                ScheduleSummaryGenerator(
                    team_identities=export_model.team_identities,
                ),
                subdirs["txt"],
                "txt",
            )
        )
    if export_model.schedules_team_csv:
        exporters.append(
            (
                TeamScheduleGenerator(
                    ga=ga,
                    team_identities=export_model.team_identities,
                ),
                subdirs["team"],
                "csv",
            ),
        )
    return tuple(exporters)


@dataclass(slots=True)
class ExportManager:
    """Manager for exporting schedules in different formats."""

    schedules: list[Schedule]
    exporters: tuple

    async def export_all(self) -> None:
        """Export all schedules to the different formats asynchronously."""
        tasks = []
        for exporter, subdir, ext in self.exporters:
            for i, sched in enumerate(self.schedules, start=1):
                name = f"front{sched.rank}_sched{i}"
                tasks.append(exporter.export(sched, subdir / f"{name}.{ext}"))
        await asyncio.gather(*tasks)


@dataclass(slots=True)
class OutputDirManager:
    """Manage creation/clearing of output directories."""

    output_dir: Path
    subdirs: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set up the output directories for the different export formats."""
        if self.output_dir.exists():
            logger.debug("Output directory %s already exists. Clearing contents.", self.output_dir)
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory: %s", self.output_dir)
        self.subdirs.update(
            {
                "csv": self.output_dir / "schedules_csv",
                "html": self.output_dir / "schedules_html",
                "txt": self.output_dir / "summary_reports",
                "team": self.output_dir / "schedules_team_csv",
            }
        )
        for sd in self.subdirs.values():
            sd.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ScheduleSummaryGenerator:
    """Exporter for generating summaries of schedules."""

    team_identities: dict[int, str]

    def get_text_summary(self, schedule: Schedule) -> tuple[str, ...]:  # noqa: PLR0915
        """Get a text summary of the schedule."""
        txt = []
        objectives = tuple(FitnessObjective)
        length_objectives = [len(name) for name in objectives]
        max_len_obj = max(length_objectives, default=0) + 1
        txt.append(f"FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)})\n")
        txt.append("\nAttributes:\n")
        txt.append("--------------------\n")
        txt.extend(
            f"{slot}: {getattr(schedule, slot)}\n"
            for slot in schedule.__slots__
            if slot not in ("schedule", "fitness", "team_fitnesses", "team_events", "team_rounds")
        )
        txt.append(f"Length: {schedule.get_size()}\n")
        txt.append("\nFitness:\n")
        txt.append("--------------------------\n")
        for name, score in zip(objectives, schedule.fitness, strict=True):
            txt.append(f"{name:<{max_len_obj}}: {score}\n")
        txt.append(f"{'-' * (max_len_obj + 15)}\n")
        txt.append(f"{'Total':<{max_len_obj}}: {schedule.fitness.sum()}\n")
        txt.append(f"{'Percentage':<{max_len_obj}}: {sum(schedule.fitness) / len(schedule.fitness):.2%}\n")
        team_fits = schedule.team_fitnesses
        min_obj = team_fits.min(axis=0)
        max_obj = team_fits.max(axis=0)
        mean_obj = team_fits.mean(axis=0)
        range_obj = max_obj - min_obj
        txt.append("\nPer-Objective Statistics (Team Distribution):\n")
        txt.append("-" * 65 + "\n")
        txt.append(f"{'Objective':<25} | {'Min':<8} | {'Max':<8} | {'Avg':<8} | {'Range':<8}\n")
        txt.append("-" * 65 + "\n")
        for i, name in enumerate(objectives):
            txt.append(
                f"{name:<25} | {min_obj[i]:<8.6f} | {max_obj[i]:<8.6f} | {mean_obj[i]:<8.6f} | {range_obj[i]:<8.6f}\n"
            )
        all_teams = schedule.ctx.teams_list
        team_fits = schedule.team_fitnesses
        total_fits = team_fits.sum(axis=1)
        max_team_f = total_fits.max()
        min_team_f = total_fits.min()
        txt.append("\nTeam fitnesses (sorted by total fitness descending):\n")
        txt.append("----------------------------------------------------\n")
        txt.append(f"Max     : {max_team_f:.6f}\n")
        txt.append(f"Min     : {min_team_f:.6f}\n")
        txt.append(f"Range   : {max_team_f - min_team_f:.6f}\n")
        txt.append(f"Average : {sum(total_fits) / len(total_fits):.6f}\n")
        objs_header = "|".join(f"{name:<{length_objectives[i] + 1}}" for i, name in enumerate(objectives))
        header = f"\n{'Team':<5}|{objs_header}|Sum\n"
        txt.append(header)
        txt.append("-" * len(header) + "\n")
        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        for t, fit in sorted(zip(all_teams, team_fits, strict=True), key=lambda x: -x[1].sum()):
            fitness_row = (
                f"{score:<{length_objectives[i] + 1}.6f}" if score > 0.000001 else f"{0:<{length_objectives[i] + 1}}"
                for i, score in enumerate(fit)
            )
            fitness_str = "|".join(fitness_row)
            if (team_id := normalized_teams[t]) == -1:
                continue
            txt.append(f"{team_id:<5}|{fitness_str}|{sum(fit):.4f}\n")
        txt.append(
            "\nTeam Events (sorted, for dev use, diff check with others to ensure truly different schedules created):\n"
        )
        txt.append("------------------------------------------------------------\n")
        team_events = [sorted(events) for events in schedule.team_events.values()]
        team_events.sort()
        for events in team_events:
            events_str = ", ".join(str(e) for e in events) + "\n"
            txt.append(events_str)
        return tuple(txt)

    async def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a text summary report for a single schedule."""
        try:
            with path.open("w", encoding="utf-8") as f:
                txt_data = self.get_text_summary(schedule)
                f.writelines(txt_data)
        except OSError:
            logger.exception("Failed to write summary report to file %s", path)


@dataclass(slots=True)
class TeamScheduleGenerator:
    """Exporter for generating team schedules."""

    ga: GA
    team_identities: dict[int, str]

    def get_team_schedule(self, schedule: Schedule) -> tuple[tuple[str, ...], ...]:
        """Get the schedule for each team."""
        config = self.ga.context.app_config.tournament
        rows = []
        headers: list[str] = ["Team"]
        for roundtype, rounds_per_team in config.roundreqs.items():
            if rounds_per_team == 1:
                headers.extend([f"{roundtype.capitalize()}", ""])
            else:
                for i in range(1, rounds_per_team + 1):
                    headers.extend([f"{roundtype.capitalize()} {i}", ""])
        rows.append(tuple(headers))
        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        team_events: dict[int, set[int]] = defaultdict(set)
        for event_id, t in enumerate(schedule.schedule):
            if t == -1:
                continue
            team_id = normalized_teams[t]
            team_events[team_id].add(event_id)
        ep = self.ga.context.event_properties
        for team_id, events in sorted(team_events.items()):
            r = [str(team_id)]
            for event_id in sorted(events):
                r.append(str(ep.timeslot[event_id]))
                r.append(str(ep.location[event_id]))
            rows.append(tuple(r))
        return tuple(rows)

    async def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a CSV file with team schedules, sorted by team IDs."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                rows = self.get_team_schedule(schedule)
                writer = csv.writer(f)
                writer.writerows(rows)
        except OSError:
            logger.exception("Failed to write team schedules to file %s", path)


@dataclass(slots=True)
class ParetoSummaryGenerator:
    """Exporter for generating Pareto front summaries."""

    def get_pareto_summary(self, pop: list[Schedule]) -> tuple[tuple[str, ...], ...]:
        """Get a summary of the Pareto front."""
        summary = []
        header = ["Schedule", "ID", "Hash", "Length", "Rank"]
        header.extend(name.value for name in FitnessObjective)
        header.extend(["Sum", "Origin", "Mutations", "Clones"])
        summary.append(tuple(header))
        for i, s in enumerate(pop, start=1):
            row = [str(i), str(id(s)), str(hash(s)), str(s.get_size()), str(s.rank)]
            row.extend(f"{score:.4f}" for score in s.fitness)
            row.append(f"{s.fitness.sum():.4f}")
            row.extend([s.origin, str(s.mutations), str(s.clones)])
            summary.append(tuple(row))
        return tuple(summary)

    def export(self, pop: list[Schedule], path: Path) -> None:
        """Generate a summary of the Pareto front."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                summary = self.get_pareto_summary(pop)
                writer = csv.writer(f)
                writer.writerows(summary)
        except OSError:
            logger.exception("Failed to write Pareto summary to file %s", path)
