"""Base class for exporting schedules."""

from abc import ABC, abstractmethod

from ..config.config import RoundType
from ..data_model.location import Location
from ..data_model.team import Individual, Team
from ..data_model.time import TimeSlot


class Exporter(ABC):
    """Abstract base class for exporting schedules."""

    @abstractmethod
    def export(self, schedule: Individual, filename: str) -> None:
        """Export the schedule to a given filename."""

    @abstractmethod
    def render_grid(self, title: str, schedule: Individual) -> list[str]:
        """Render a schedule grid for a specific round type."""

    def _group_by_type(self, schedule: Individual) -> dict[RoundType, Individual]:
        """Group the schedule by round type."""
        grouped = {}
        for event, team in sorted(schedule.items(), key=lambda item: (item[0].timeslot.start)):
            grouped.setdefault(event.round_type, {})
            grouped[event.round_type][event] = team
        return grouped


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
