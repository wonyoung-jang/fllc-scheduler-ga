"""Base class for exporting schedules."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..data_model.event import EventProperties
    from ..data_model.location import Location
    from ..data_model.schedule import Schedule
    from ..data_model.time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleExporter(ABC):
    """Abstract base class for exporting schedules."""

    time_fmt: str
    event_properties: EventProperties

    def export(self, schedule: Schedule, path: Path) -> None:
        """Export the schedule to a given filename."""
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return

        schedule_by_type = self._group_by_type(schedule)

        try:
            self.write_to_file(schedule_by_type, path)
            logger.debug("Schedule successfully exported to %s", path)
        except OSError:
            logger.exception("Failed to export schedule to %s", path)

    def _group_by_type(self, schedule: Schedule) -> dict[str, dict[int, int]]:
        """Group the schedule by round type."""
        grouped = {}
        normalized_teams = schedule.normalized_teams()
        for event, team in enumerate(schedule.schedule):
            if team == -1:
                continue

            rt = self.event_properties.roundtype[event]
            grouped.setdefault(rt, {})
            grouped[rt][event] = normalized_teams.get(team)
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

    @abstractmethod
    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""

    @abstractmethod
    def render_grid(self, schedule_by_type: dict[str, dict[int, int]]) -> Iterator[str | Iterator[str]]:
        """Render a schedule grid for a specific round type."""


@dataclass(slots=True)
class CsvScheduleExporter(ScheduleExporter):
    """Exporter for schedules in CSV format."""

    def render_grid(self, schedule_by_type: dict[str, dict[int, int]]) -> Iterator[list[str]]:
        """Write a single schedule grid to a CSV writer."""
        for title, schedule in schedule_by_type.items():
            yield [title]
            if not schedule:
                yield ["No events scheduled for this round type.", []]
                continue

            timeslots, locations, grid_lookup = self._build_grid_data(schedule)
            yield ["Time"] + [str(loc) for loc in locations]
            for ts in timeslots:
                row = [ts.start.strftime(self.time_fmt)]
                for loc in locations:
                    team = grid_lookup.get((ts, loc))
                    row.append(team)
                yield row
            yield []

    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerows(self.render_grid(schedule_by_type))


@dataclass(slots=True)
class HtmlScheduleExporter(ScheduleExporter):
    """Exporter for schedules in HTML format."""

    def render_grid(self, schedule_by_type: dict[str, dict[int, int]]) -> Iterator[str]:
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
                    if team is None:
                        yield "<td></td>"
                    else:
                        yield f"<td>{team}</td>"
                yield "</tr>"
            yield "</tbody></table>"

    def write_to_file(self, schedule_by_type: dict[str, dict[int, int]], filename: Path) -> None:
        """Write the schedule to a file."""
        with filename.open("w", encoding="utf-8") as f:
            f.write("".join(self.render_grid(schedule_by_type)))
