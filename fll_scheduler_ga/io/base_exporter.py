"""Base class for exporting schedules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..data_model.event import Event, EventFactory, EventProperties
    from ..data_model.location import Location
    from ..data_model.schedule import Schedule
    from ..data_model.time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class Exporter(ABC):
    """Abstract base class for exporting schedules."""

    time_fmt: str
    event_factory: EventFactory
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

    def _group_by_type(self, schedule: Schedule) -> dict[str, dict[Event, int]]:
        """Group the schedule by round type."""
        grouped = {}
        normalized_teams = schedule.normalized_teams()
        event_map = self.event_factory.as_mapping()
        for i, team in enumerate(schedule.schedule):
            if team == -1:
                continue

            event = event_map.get(i)
            if not event:
                continue

            grouped.setdefault(event.roundtype, {})
            grouped[event.roundtype][event] = normalized_teams.get(team)
        return grouped

    def _build_grid_data(
        self, schedule: dict[Event, int]
    ) -> tuple[list[TimeSlot], list[Location], dict[tuple[TimeSlot, Location], int]]:
        """Build the common grid data structure from a schedule."""
        grid_lookup = {(e.timeslot, e.location): team for e, team in schedule.items()}
        timeslots = sorted(
            {i[0] for i in grid_lookup},
            key=lambda ts: ts.start,
        )
        locations = sorted(
            {i[1] for i in grid_lookup},
            key=lambda loc: (
                loc.name,
                loc.side if loc.side else 0,
            ),
        )
        return timeslots, locations, grid_lookup

    @abstractmethod
    def write_to_file(self, schedule_by_type: dict[str, dict[Event, int]], filename: Path) -> None:
        """Write the schedule to a file."""

    @abstractmethod
    def render_grid(self, schedule_by_type: dict[str, dict[Event, int]]) -> Iterator[str | Iterator[str]]:
        """Render a schedule grid for a specific round type."""
