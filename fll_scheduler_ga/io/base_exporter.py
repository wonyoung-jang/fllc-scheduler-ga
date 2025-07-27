"""Base class for exporting schedules."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config.config import RoundType
from ..data_model.location import Location
from ..data_model.team import Team
from ..data_model.time import TimeSlot
from ..genetic.schedule import Individual, Schedule

logger = logging.getLogger(__name__)


class Exporter(ABC):
    """Abstract base class for exporting schedules."""

    def export(self, schedule: Schedule, filename: Path) -> None:
        """Export the schedule to a given filename."""
        if not schedule:
            logger.warning("Cannot export an empty schedule.")
            return

        schedule_by_type = self._group_by_type(schedule)

        try:
            self.write_to_file(schedule_by_type, filename)
            logger.debug("Schedule successfully exported to %s", filename)
        except OSError:
            logger.exception("Failed to export schedule to %s", filename)

    def _group_by_type(self, schedule: Schedule) -> dict[RoundType, Individual]:
        """Group the schedule by round type."""
        grouped = {}
        normalized_teams = schedule.normalize_teams()
        for event, team in sorted(schedule.items(), key=lambda item: (item[0].identity)):
            grouped.setdefault(event.round_type, {})
            grouped[event.round_type][event] = normalized_teams.get(team)
        return grouped

    @abstractmethod
    def write_to_file(self, schedule_by_type: dict[RoundType, Individual], filename: Path) -> None:
        """Write the schedule to a file."""

    @abstractmethod
    def render_grid(self, title: str, schedule: Schedule) -> list[str]:
        """Render a schedule grid for a specific round type."""


class GridBasedExporter(Exporter):
    """Base class for exporters that render a grid-based schedule."""

    def _build_grid_data(
        self, schedule: Schedule
    ) -> tuple[list[TimeSlot], list[Location], dict[tuple[TimeSlot, Location], Team]]:
        """Build the common grid data structure from a schedule."""
        grid_lookup = {(e.timeslot, e.location): team for e, team in schedule.items()}
        timeslots = sorted(
            {i[0] for i in grid_lookup},
            key=lambda ts: ts.start,
        )
        locations = sorted(
            {i[1] for i in grid_lookup},
            key=lambda loc: (
                loc.identity,
                loc.side if hasattr(loc, "side") else 0,
            ),
        )
        return timeslots, locations, grid_lookup
