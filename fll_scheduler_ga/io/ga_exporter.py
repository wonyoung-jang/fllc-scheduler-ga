"""Module for exporting schedules to different formats."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import FitnessObjective
from .plot import Plot
from .schedule_exporter import CsvScheduleExporter, HtmlScheduleExporter, normalize_teams

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import ExportModel
    from ..data_model.event import EventProperties
    from ..data_model.schedule import Schedule
    from ..genetic.ga import GA

logger = getLogger(__name__)


def generate_summary(ga: GA, output_dir: Path, export_model: ExportModel) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    subdirs = OutputDirManager(output_dir).subdirs
    total_pop = ga.total_population
    if not export_model.no_plotting and total_pop:
        Plot(
            ga=ga,
            save_dir=output_dir,
            objectives=list(FitnessObjective),
            ref_points=ga.context.nsga3.refs.points,
            export_model=export_model,
        ).plot()

    schedules = ga.pareto_front() if export_model.front_only else total_pop
    schedules.sort(key=lambda s: (s.rank, -sum(s.fitness)))

    time_fmt = ga.context.app_config.tournament.time_fmt
    event_properties = ga.context.event_properties
    export_manager = ExportManager(schedules, subdirs, time_fmt, event_properties, ga, export_model)
    export_manager.export_all()

    if export_model.pareto_summary:
        pareto_summary_gen = ParetoSummaryGenerator()
        pareto_summary_gen.export(total_pop, output_dir / "pareto_summary.csv")


@dataclass(slots=True)
class ExportManager:
    """Manager for exporting schedules in different formats."""

    schedules: list[Schedule]
    subdirs: dict[str, Path]
    time_fmt: str
    event_properties: EventProperties
    ga: GA
    export_model: ExportModel

    def get_exporters(self) -> list:
        """Get the list of exporters based on the export model."""
        exporters = []
        if self.export_model.schedules_csv:
            exporters.append(
                (
                    CsvScheduleExporter(
                        time_fmt=self.time_fmt,
                        team_identities=self.export_model.team_identities,
                        event_properties=self.event_properties,
                    ),
                    self.subdirs["csv"],
                    "csv",
                )
            )
        if self.export_model.schedules_html:
            exporters.append(
                (
                    HtmlScheduleExporter(
                        time_fmt=self.time_fmt,
                        team_identities=self.export_model.team_identities,
                        event_properties=self.event_properties,
                    ),
                    self.subdirs["html"],
                    "html",
                )
            )
        if self.export_model.summary_reports:
            exporters.append(
                (
                    ScheduleSummaryGenerator(
                        team_identities=self.export_model.team_identities,
                    ),
                    self.subdirs["txt"],
                    "txt",
                )
            )
        if self.export_model.schedules_team_csv:
            exporters.append(
                (
                    TeamScheduleGenerator(
                        ga=self.ga,
                        team_identities=self.export_model.team_identities,
                    ),
                    self.subdirs["team"],
                    "csv",
                ),
            )
        return exporters

    def export_all(self) -> None:
        """Export all schedules to the different formats."""
        exporters = self.get_exporters()
        for exporter, subdir, ext in exporters:
            for i, sched in enumerate(self.schedules, start=1):
                name = f"front{sched.rank}_sched{i}"
                exporter.export(sched, subdir / f"{name}.{ext}")


@dataclass(slots=True)
class OutputDirManager:
    """Manage creation/clearing of output directories."""

    output_dir: Path
    subdirs: dict[str, Path] = None

    def __post_init__(self) -> None:
        """Set up the output directories for the different export formats."""
        if self.output_dir.exists():
            logger.debug("Output directory %s already exists. Clearing contents.", self.output_dir)
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory: %s", self.output_dir)

        self.subdirs = {
            "csv": self.output_dir / "schedules_csv",
            "html": self.output_dir / "schedules_html",
            "txt": self.output_dir / "summary_reports",
            "team": self.output_dir / "schedules_team_csv",
        }
        for sd in self.subdirs.values():
            sd.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ScheduleSummaryGenerator:
    """Exporter for generating summaries of schedules."""

    team_identities: dict[int, str]

    def get_text_summary(self, schedule: Schedule) -> list[str]:  # noqa: PLR0915
        """Get a text summary of the schedule."""
        txt = []
        objectives = list(FitnessObjective)
        len_objectives = [len(name) for name in objectives]
        max_len_obj = max(len_objectives, default=0) + 1
        txt.append(f"FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)})\n")

        txt.append("\nAttributes:\n")
        txt.append("--------------------\n")
        txt.extend(
            f"{slot}: {getattr(schedule, slot)}\n"
            for slot in schedule.__slots__
            if slot not in ("schedule", "fitness", "team_fitnesses", "team_events", "team_rounds")
        )
        txt.append(f"Length: {len(schedule)}\n")

        txt.append("\nFitness:\n")
        txt.append("--------------------------\n")
        for name, score in zip(objectives, schedule.fitness, strict=True):
            txt.append(f"{name:<{max_len_obj}}: {score:.6f}\n")
        txt.append(f"{'-' * (max_len_obj + 15)}\n")
        txt.append(f"{'Total':<{max_len_obj}}: {sum(schedule.fitness):.6f}\n")
        txt.append(f"{'Percentage':<{max_len_obj}}: {sum(schedule.fitness) / len(schedule.fitness):.2%}\n")

        team_fits = schedule.team_fitnesses
        min_obj = team_fits.min(axis=0)
        max_obj = team_fits.max(axis=0)
        mean_obj = team_fits.mean(axis=0)

        txt.append("\nPer-Objective Statistics (Team Distribution):\n")
        txt.append("-" * 65 + "\n")
        txt.append(f"{'Objective':<25} | {'Min':<8} | {'Max':<8} | {'Avg':<8}\n")
        txt.append("-" * 65 + "\n")

        for i, name in enumerate(objectives):
            txt.append(f"{name:<25} | {min_obj[i]:<8.4f} | {max_obj[i]:<8.4f} | {mean_obj[i]:<8.4f}\n")

        all_teams = schedule.teams
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

        objs_header = "|".join(f"{name:<{len_objectives[i] + 1}}" for i, name in enumerate(objectives))
        header = f"\n{'Team':<5}|{objs_header}|Sum\n"
        txt.append(header)
        txt.append("-" * len(header) + "\n")

        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        for t, fit in sorted(zip(all_teams, team_fits, strict=True), key=lambda x: -x[1].sum()):
            fitness_row = (f"{score:<{len_objectives[i] + 1}.4f}" for i, score in enumerate(fit))
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

        return txt

    def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a text summary report for a single schedule."""
        try:
            txt_data = self.get_text_summary(schedule)
            with path.open("w", encoding="utf-8") as f:
                f.writelines(txt_data)
        except OSError:
            logger.exception("Failed to write summary report to file %s", path)


@dataclass(slots=True)
class TeamScheduleGenerator:
    """Exporter for generating team schedules."""

    ga: GA
    team_identities: dict[int, str]

    def get_team_schedule(self, schedule: Schedule) -> list[list[str]]:
        """Get the schedule for each team."""
        config = self.ga.context.app_config.tournament
        rows: list[list[str]] = []
        headers: list[str] = ["Team"]

        for roundtype, rounds_per_team in config.roundreqs.items():
            if rounds_per_team == 1:
                headers.extend([f"{roundtype.capitalize()}", ""])
            else:
                for i in range(1, rounds_per_team + 1):
                    headers.extend([f"{roundtype.capitalize()} {i}", ""])

        rows.append(headers)

        normalized_teams = normalize_teams(schedule.schedule, self.team_identities)
        team_events: dict[int, set[int]] = defaultdict(set)
        for event_id, t in enumerate(schedule):
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
            rows.append(r)
        return rows

    def export(self, schedule: Schedule, path: Path) -> None:
        """Generate a CSV file with team schedules, sorted by team IDs."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                rows = self.get_team_schedule(schedule)
                csv.writer(f).writerows(rows)
        except OSError:
            logger.exception("Failed to write team schedules to file %s", path)


@dataclass(slots=True)
class ParetoSummaryGenerator:
    """Exporter for generating Pareto front summaries."""

    def get_pareto_summary(self, pop: list[Schedule]) -> list[list[str]]:
        """Get a summary of the Pareto front."""
        summary: list[list[str]] = []
        header = ["Schedule", "ID", "Hash", "Length", "Rank"]
        header.extend(name.value for name in FitnessObjective)
        header.extend(["Sum", "Origin", "Mutations", "Clones"])
        summary.append(header)

        for i, s in enumerate(pop, start=1):
            row = [str(i), str(id(s)), str(hash(s)), str(len(s)), str(s.rank)]
            row.extend(f"{score:.4f}" for score in s.fitness)
            row.append(f"{s.fitness.sum():.4f}")
            row.extend([s.origin, str(s.mutations), str(s.clones)])
            summary.append(row)
        return summary

    def export(self, pop: list[Schedule], path: Path) -> None:
        """Generate a summary of the Pareto front."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                summary = self.get_pareto_summary(pop)
                writer.writerows(summary)
        except OSError:
            logger.exception("Failed to write Pareto summary to file %s", path)
