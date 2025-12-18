"""Module for exporting schedules in various formats."""

from __future__ import annotations

import csv
import html
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..data_model.event import EventProperties
    from ..data_model.location import Location
    from ..data_model.schedule import Schedule
    from ..data_model.timeslot import TimeSlot

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
    async def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""

    @abstractmethod
    def render_grid(self, schedule_dict: dict[int, int]) -> Iterator[str | list[str]]:
        """Render a schedule grid for a specific round type."""


@dataclass(slots=True)
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


@dataclass(slots=True)
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
