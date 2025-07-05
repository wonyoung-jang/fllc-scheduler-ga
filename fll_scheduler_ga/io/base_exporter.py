"""Base class for exporting schedules."""

from abc import ABC, abstractmethod

from ..config.config import RoundType
from ..data_model.team import Individual


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
