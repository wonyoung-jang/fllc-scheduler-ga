"""Module for exporting schedules to different formats."""

import csv
import html
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from fll_scheduler_ga.constants import FitnessObjective

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from fll_scheduler_ga.domain.model import EventProperties, Location, Schedule, TimeSlot

logger = getLogger(__name__)
mpl_logger = logging.getLogger("visualize.plot")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.style.use("seaborn-v0_8-whitegrid")


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
class SummaryManager:
    """Manages the generation and export of summary reports for GA runs."""

    outdir: Path
    plot: Visualizer | None
    pareto_front: list[Schedule]
    total_population: list[Schedule]
    roundreqs: dict[str, int]
    time_fmt: str
    team_ids: dict[int, str]
    evt_prop: EventProperties
    export_pareto_summary: bool
    export_schedules_csv: bool
    export_schedules_html: bool
    export_summary_reports: bool
    export_schedules_team_csv: bool
    export_front_only: bool
    subdirs: dict[str, Path] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize subdirectories for exports."""
        self.subdirs = self.get_subdirs()

    def get_subdirs(self) -> dict[str, Path]:
        """Get the subdirectories for different export formats."""
        logger.debug("Output directory: %s", self.outdir)
        subdirs = {
            "csv": self.outdir / "schedules_csv",
            "html": self.outdir / "schedules_html",
            "txt": self.outdir / "summary_reports",
            "team": self.outdir / "schedules_team_csv",
        }
        for sd in subdirs.values():
            sd.mkdir(parents=True, exist_ok=True)
        return subdirs

    def generate(self) -> None:
        """Generate summary reports for the GA run."""
        self.export_all()
        if self.plot is not None:
            self.plot.plot()
        if self.export_pareto_summary:
            self.export_pareto(self.outdir / "pareto_summary.csv")

    def get_exporters(self) -> Iterator:
        """Get the list of exporters based on the export model."""
        if self.export_schedules_csv:
            yield (CsvScheduleExporter(self.time_fmt, self.team_ids, self.evt_prop), self.subdirs["csv"], "csv")
        if self.export_schedules_html:
            yield (HtmlScheduleExporter(self.time_fmt, self.team_ids, self.evt_prop), self.subdirs["html"], "html")
        if self.export_summary_reports:
            yield ScheduleSummaryGenerator(self.team_ids), self.subdirs["txt"], "txt"
        if self.export_schedules_team_csv:
            yield (TeamScheduleGenerator(self.roundreqs, self.team_ids, self.evt_prop), self.subdirs["team"], "csv")

    def export_all(self) -> None:
        """Export all schedules to different formats."""
        schedules = self.pareto_front if self.export_front_only else self.total_population
        schedules.sort(key=lambda s: (s.rank, -sum(s.fitness)))
        for exporter, subdir, ext in self.get_exporters():
            for i, sched in enumerate(schedules, start=1):
                exporter.export(sched, subdir / f"front{sched.rank}_sched{i}.{ext}")

    def get_pareto_summary(self) -> Iterator[list[str]]:
        """Get a summary of the Pareto front."""
        header = ["Schedule", "ID", "Hash", "Length", "Rank", *(n.value for n in FitnessObjective)]
        header.extend(["Sum", "Origin", "Mutations", "Clones"])
        yield header
        for i, s in enumerate(self.total_population, start=1):
            row = [str(i), str(id(s)), str(hash(s)), str(s.get_size()), str(s.rank)]
            row.extend(f"{score:.4f}" for score in s.fitness)
            row.extend([f"{s.fitness.sum():.4f}", s.origin, str(s.mutations), str(s.clones)])
            yield row

    def export_pareto(self, path: Path) -> None:
        """Generate a summary of the Pareto front."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(self.get_pareto_summary())
        except OSError:
            logger.exception("Failed to write Pareto summary to file %s", path)


@dataclass(slots=True)
class ScheduleExporter(ABC):
    """Abstract base class for exporting schedules."""

    time_fmt: str
    team_identities: dict[int, str]
    event_properties: EventProperties

    def export(self, schedule: Schedule, path: Path) -> None:
        """Export the schedule to a given filename."""
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return
        try:
            self.write_to_file(self._group_by_type(schedule), path)
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
        timeslots: list[TimeSlot] = sorted({i[0] for i in grid_lookup}, key=lambda ts: ts.start)
        locations: list[Location] = sorted(
            {i[1] for i in grid_lookup}, key=lambda loc: (loc.name, loc.side if loc.side != -1 else 0)
        )
        return timeslots, locations, grid_lookup

    def get_table_data(self, schedule_dict: dict[int, int]) -> list[list[str]]:
        """Generate a 2D matrix of strings representing the grid for a single round type."""
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
    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None: ...


class CsvScheduleExporter(ScheduleExporter):
    """Exporter for schedules in CSV format."""

    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[list[str]]:
        """Write a single schedule grid as CSV rows."""
        if not schedule_dict:
            yield ["No events scheduled for this round type."]
        else:
            yield from self.get_table_data(schedule_dict)
        yield []

    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for title, schedule_dict in schedule_by_type.items():
                writer.writerow([title])
                writer.writerows(self.render_grid(schedule_dict))


_HTML_START: str = """
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
_HTML_END: str = """
    </div>
</body>
</html>
"""


class HtmlScheduleExporter(ScheduleExporter):
    """Exporter for schedules in HTML format."""

    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[str]:
        """Render a single schedule grid as an HTML table."""
        if not schedule_dict:
            yield "<p>No events scheduled.</p>"
        else:
            data = self.get_table_data(schedule_dict)
            yield "<table>"
            yield "<thead><tr>"
            for cell in data[0]:
                yield f"<th>{html.escape(cell)}</th>"
            yield "</tr></thead>"
            yield "<tbody>"
            for row in data[1:]:
                yield "<tr>"
                for cell in row:
                    # First column is time, others are locations/teams
                    yield f"<td>{html.escape(cell)}</td>"
                yield "</tr>"
            yield "</tbody>"
            yield "</table>"

    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", encoding="utf-8") as f:
            f.write(_HTML_START)
            for title, schedule_dict in schedule_by_type.items():
                f.write(f"<h2>{html.escape(title)}</h2>")
                f.write("".join(self.render_grid(schedule_dict)))
            f.write(_HTML_END)


@dataclass(slots=True)
class ScheduleSummaryGenerator:
    """Exporter for generating summaries of schedules."""

    team_identities: dict[int, str]

    def get_text_summary(self, schedule: Schedule) -> Iterator[str]:
        """Get a text summary of the schedule."""
        obj = tuple(FitnessObjective)
        objname_len = [len(name) for name in obj]
        objname_len_max = max(objname_len, default=0) + 1
        team_fits = schedule.team_fitnesses
        min_obj = team_fits.min(axis=0)
        max_obj = team_fits.max(axis=0)
        mean_obj = team_fits.mean(axis=0)
        range_obj = max_obj - min_obj
        total_fits = team_fits.sum(axis=1)
        min_team_f = total_fits.min()
        max_team_f = total_fits.max()
        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        yield f"FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)})\n"
        yield "\nAttributes:\n---\n"
        yield from (
            f"{slot}: {getattr(schedule, slot)}\n"
            for slot in schedule.__slots__
            if slot not in ("schedule", "fitness", "team_fitnesses", "team_events", "team_rounds")
        )
        yield f"Length: {schedule.get_size()}\n"
        yield "\nFitness:\n---\n"
        for name, score in zip(obj, schedule.fitness, strict=True):
            yield f"{name:<{objname_len_max}}: {score}\n"
        yield f"{'Total':<{objname_len_max}}: {schedule.fitness.sum()}\n"
        yield f"{'Percentage':<{objname_len_max}}: {sum(schedule.fitness) / len(schedule.fitness):.2%}\n"
        yield "\nPer-Objective Statistics (Team Distribution):\n---\n"
        yield f"{'Objective':<25} | {'Min':<8} | {'Max':<8} | {'Avg':<8} | {'Range':<8}\n---\n"
        for i, name in enumerate(obj):
            yield (
                f"{name:<25} | {min_obj[i]:<8.6f} | {max_obj[i]:<8.6f} | {mean_obj[i]:<8.6f} | {range_obj[i]:<8.6f}\n"
            )
        yield "\nTeam fitnesses (sorted by total fitness descending):\n---\n"
        yield f"Max     : {max_team_f:.6f}\n"
        yield f"Min     : {min_team_f:.6f}\n"
        yield f"Range   : {max_team_f - min_team_f:.6f}\n"
        yield f"Average : {sum(total_fits) / len(total_fits):.6f}\n"
        yield f"\n{'Team':<5}|{'|'.join(f'{name:<{objname_len[i] + 1}}' for i, name in enumerate(obj))}|Sum\n---\n"
        for t, fit in sorted(zip(schedule.ctx.teams_list, team_fits, strict=True), key=lambda x: -x[1].sum()):
            if (team_id := normalized_teams[t]) == -1:
                continue
            fitness_str = "|".join(
                f"{score:<{objname_len[i] + 1}.6f}" if score > 0.000001 else f"{0:<{objname_len[i] + 1}}"
                for i, score in enumerate(fit)
            )
            yield f"{team_id:<5}|{fitness_str}|{sum(fit):.4f}\n"
        yield "\nTeam Events (sorted, for devs, check diff with others to ensure schedules different):\n---\n"
        for events in sorted(sorted(e) for e in schedule.team_events.values()):
            yield f"{', '.join(str(e) for e in events)}\n"

    def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a text summary report for a single schedule."""
        try:
            with path.open("w", encoding="utf-8") as f:
                f.writelines(self.get_text_summary(schedule))
        except OSError:
            logger.exception("Failed to write summary report to file %s", path)


@dataclass(slots=True)
class TeamScheduleGenerator:
    """Exporter for generating team schedules."""

    roundreqs: dict[str, int]
    team_identities: dict[int, str]
    event_properties: EventProperties

    def get_team_schedule(self, schedule: Schedule) -> Iterator[list[str]]:
        """Get the schedule for each team."""
        header = ["Team"]
        for roundtype, rounds_per_team in self.roundreqs.items():
            if rounds_per_team == 1:
                header.extend([f"{roundtype.capitalize()}", ""])
            else:
                for i in range(1, rounds_per_team + 1):
                    header.extend([f"{roundtype.capitalize()} {i}", ""])
        yield header
        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        team_events: dict[int, set[int]] = defaultdict(set)
        for event_id, t in enumerate(schedule.schedule):
            if t == -1:
                continue
            team_id = normalized_teams[t]
            team_events[team_id].add(event_id)
        for team_id, events in sorted(team_events.items()):
            row = [str(team_id)]
            for event_id in sorted(events):
                row.append(str(self.event_properties.timeslot[event_id]))
                row.append(str(self.event_properties.location[event_id]))
            yield row

    def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a CSV file with team schedules, sorted by team IDs."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(self.get_team_schedule(schedule))
        except OSError:
            logger.exception("Failed to write team schedules to file %s", path)


class Visualizer(ABC):
    """Abstract base class for visualizers."""

    @abstractmethod
    def plot(self) -> None: ...


@dataclass(slots=True)
class MatplotlibVisualizer(Visualizer):
    """A class for creating and managing plots related to the GA run."""

    total_population: list[Schedule]
    fitness_history: np.ndarray
    save_dir: str | Path | None
    ref_points: np.ndarray
    cmap_name: str
    is_plot_fitness: bool
    is_plot_parallel: bool
    is_plot_scatter: bool

    def plot(self) -> None:
        """Create all plots."""
        if self.is_plot_fitness:
            self.plot_fitness()
        if self.is_plot_parallel:
            self.plot_parallel()
        if self.is_plot_scatter:
            self.plot_scatter()

    def plot_fitness(self) -> None:
        """Create figure that summarizes how the average fitness of the first Pareto front evolved by generation."""
        # Filter out generations (if program terminated early)
        history = self.fitness_history[self.fitness_history[:, 0] >= 0]
        if not history.any():
            mpl_logger.error("Cannot plot fitness. No generation history was recorded.")
            return
        fig, ax = plt.subplots(figsize=(12, 7))
        columns = list(FitnessObjective)
        x = np.arange(history.shape[0])
        for i, col in enumerate(columns):
            y = history[:, i]
            ax.plot(x, y, linewidth=2.5, alpha=0.9, label=col)
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle="--", linewidth=0.8, label=f"{col} Trend (deg={len(z) - 1})")
        ax.set(title="Fitness over time", xlabel="Generations", ylabel="Average fitnesses")
        ax.legend(title="Objectives", fontsize=10)
        fig.tight_layout()
        self._finalize(fig, "fitness_vs_generation.png")

    def plot_parallel(self) -> None:
        """Create the parallel coordinates plot."""
        data = np.array([p.fitness for p in self.total_population])
        ranks = np.array([p.rank for p in self.total_population], dtype=int)
        fig, ax = plt.subplots(figsize=(12, 7))
        x = range(len(FitnessObjective))
        colors = plt.get_cmap(self.cmap_name)(np.linspace(0, 1, len(self.total_population)))
        for i, ind_fitness in enumerate(data):
            ax.plot(x, ind_fitness, color=colors[i], alpha=0.7, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(FitnessObjective, rotation=15, ha="right")
        ax.set(title="Trade-off parallel coordinates", xlabel="Objectives", ylabel="Score")
        plt.xticks(rotation=15, ha="right")
        self._attach_colorbar(ax, ranks, label="Rank")
        self._finalize(fig, "pareto_parallel.png")

    def plot_scatter(self) -> None:
        """Create a 2D or 3D scatter plot of the Pareto front."""
        n_obj = len(FitnessObjective)
        data = np.array([p.fitness for p in self.total_population])
        ranks = np.array([p.rank for p in self.total_population], dtype=int)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        x_obj, y_obj, z_obj = FitnessObjective
        ax.view_init(azim=45, elev=40)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=60, c=ranks, cmap=self.cmap_name)
        ax.set(
            title=f"{n_obj}D scatter plot of schedules", xlabel=x_obj, ylabel=y_obj, zlabel=z_obj, box_aspect=[1, 1, 1]
        )
        self._attach_colorbar(ax, ranks, label="Rank")
        self._finalize(fig, f"pareto_scatter_{n_obj}d.png")
        ax.scatter(
            self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2], s=30, c="red", label="Reference Points"
        )
        ax.legend()
        self._finalize(fig, f"pareto_scatter_{n_obj}d_ref.png")

    def _finalize(self, fig: Figure, filename: str) -> None:
        """Finalize the plot by saving or showing it."""
        try:
            if self.save_dir:
                path = Path(self.save_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=300)
                mpl_logger.debug("Saved plot: %s", path)
            else:
                plt.show()
        except Exception:
            mpl_logger.exception("Error saving plot to %s", path)
        finally:
            plt.close(fig)

    def _attach_colorbar(
        self, ax: Axes, values: np.ndarray[tuple[int, ...], np.dtype[Any]], label: str | None = None
    ) -> None:
        """Attach a colorbar to the given axes."""
        unique_values = sorted(set(values))
        if len(unique_values) <= 10:
            cmap = plt.get_cmap(self.cmap_name, len(unique_values))
            norm = mcolors.BoundaryNorm(np.arange(min(values) - 0.5, max(values) + 1.5), cmap.N)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(mappable=sm, ax=ax, ticks=unique_values)
        else:
            norm = plt.Normalize(min(values), max(values))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap_name)
            cbar = plt.colorbar(mappable=sm, ax=ax)
        sm.set_array([])
        if label:
            cbar.set_label(label, fontsize=12)
