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
from .schedule_exporter import CsvScheduleExporter, HtmlScheduleExporter

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import ExportModel
    from ..data_model.event import EventProperties
    from ..data_model.schedule import Schedule
    from ..genetic.ga import GA

logger = getLogger(__name__)


def generate_summary(
    ga: GA,
    output_dir: Path,
    export_model: ExportModel,
) -> None:
    """Run the fll-scheduler-ga application and generate summary reports."""
    subdirs = OutputDirManager(output_dir).subdirs
    total_pop = ga.total_population
    if not export_model.no_plotting and total_pop:
        Plot(
            ga=ga,
            save_dir=output_dir,
            cmap_name=export_model.cmap_name,
            objectives=list(FitnessObjective),
            ref_points=ga.context.nsga3.ref_points,
        ).plot()

    schedules = ga.pareto_front() if export_model.front_only else total_pop
    schedules.sort(key=lambda s: (s.rank, -sum(s.fitness)))

    time_fmt = ga.context.app_config.tournament.time_fmt
    event_properties = ga.context.event_properties
    export_manager = ExportManager(schedules, subdirs, time_fmt, event_properties, ga)
    export_manager.export_all()

    if export_model.pareto_summary:
        pareto_summary_gen = ParetoSummaryGenerator()
        pareto_summary_gen.generate(total_pop, output_dir / "pareto_summary.csv")


@dataclass(slots=True)
class ExportManager:
    """Manager for exporting schedules in different formats."""

    schedules: list[Schedule]
    subdirs: dict[str, Path]
    time_fmt: str
    event_properties: EventProperties
    ga: GA

    def export_all(self) -> None:
        """Export all schedules to the different formats."""
        csv_exporter = CsvScheduleExporter(self.time_fmt, self.event_properties)
        html_exporter = HtmlScheduleExporter(self.time_fmt, self.event_properties)

        csv_dir = self.subdirs["csv"]
        html_dir = self.subdirs["html"]
        txt_dir = self.subdirs["txt"]
        team_sched_dir = self.subdirs["team"]

        summary_gen = ScheduleSummaryGenerator()
        team_sched_gen = TeamScheduleGenerator(self.ga)

        for i, sched in enumerate(self.schedules, start=1):
            name = f"fr_{sched.rank}_sched_{i}"
            csv_exporter.export(sched, csv_dir / f"{name}.csv")
            html_exporter.export(sched, html_dir / f"{name}.html")
            summary_gen.generate(sched, txt_dir / f"{name}_summary.txt")
            team_sched_gen.generate(sched.schedule, team_sched_dir / f"{name}_team.csv")


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

    def generate(self, schedule: Schedule, path: Path) -> None:
        """Generate a text summary report for a single schedule."""
        try:
            with path.open("w", encoding="utf-8") as f:
                objectives = list(FitnessObjective)
                len_objectives = [len(name) for name in objectives]
                max_len_obj = max(len_objectives, default=0) + 1
                f.write(f"FLL Scheduler GA Summary Report (ID: {id(schedule)} | Hash: {hash(schedule)})\n")

                f.write("\n")
                f.write("Attributes:\n")
                f.write("--------------------\n")
                slots = (s for s in schedule.__slots__)
                for slot in slots:
                    f.write(f"{slot}: {getattr(schedule, slot)}\n")
                f.write(f"Length: {len(schedule)}\n")

                f.write("\n")
                f.write("Fitness:\n")
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
                for t, fit in sorted(zip(all_teams, team_fits, strict=True), key=lambda x: -x[1].sum()):
                    fitness_row = (f"{score:<{len_objectives[i] + 1}.4f}" for i, score in enumerate(fit))
                    fitness_str = "|".join(fitness_row)
                    team_id = normalized_teams[t]
                    if team_id is None:
                        team_id = -1
                    f.write(f"{team_id:<5}|{fitness_str}|{sum(fit):.4f}\n")
        except OSError:
            logger.exception("Failed to write summary report to file %s", path)


@dataclass(slots=True)
class TeamScheduleGenerator:
    """Exporter for generating team schedules."""

    ga: GA

    def generate(self, schedule: Schedule, path: Path) -> None:
        """Generate a CSV file with team schedules, sorted by team IDs."""
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                config = self.ga.context.app_config.tournament
                rows: list[list[str]] = []
                headers: list[str] = ["Team"]

                for r in config.rounds:
                    roundtype = r.roundtype.capitalize()
                    rounds_per_team = r.rounds_per_team
                    if rounds_per_team == 1:
                        headers.append(f"{roundtype}")
                        headers.append("")
                    else:
                        for i in range(1, rounds_per_team + 1):
                            headers.append(f"{roundtype} {i}")
                            headers.append("")

                rows.append(headers)

                team_events: dict[int, set[int]] = defaultdict(set)
                for event_id, team_id in enumerate(schedule):
                    if team_id >= 0:
                        team_events[team_id].add(event_id)

                ep = self.ga.context.event_properties
                for team_id, events in sorted(team_events.items()):
                    r = [str(team_id + 1)]
                    for event_id in sorted(events):
                        r.append(str(ep.timeslot[event_id]))
                        r.append(str(ep.location[event_id]))
                    rows.append(r)

                csv.writer(f).writerows(rows)
        except OSError:
            logger.exception("Failed to write team schedules to file %s", path)


@dataclass(slots=True)
class ParetoSummaryGenerator:
    """Exporter for generating Pareto front summaries."""

    def generate(self, pop: list[Schedule], path: Path) -> None:
        """Generate a summary of the Pareto front."""
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write("Schedule,ID,Hash,Rank,")
                for name in FitnessObjective:
                    f.write(f"{name.value},")
                f.write("Sum,Origin,Mutations,Clones\n")

                for i, s in enumerate(pop, start=1):
                    f.write(f"{i},{id(s)},{hash(s)},{s.rank},")
                    for score in s.fitness:
                        f.write(f"{score:.4f},")
                    f.write(f"{s.fitness.sum():.4f},{s.origin},{s.mutations},{s.clones}\n")
        except OSError:
            logger.exception("Failed to write Pareto summary to file %s", path)
