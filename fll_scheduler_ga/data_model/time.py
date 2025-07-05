"""Time data module for FLL Scheduler GA."""

from dataclasses import dataclass
from datetime import datetime

HHMM_FMT = "%H:%M"


@dataclass(slots=True, frozen=True)
class TimeSlot:
    """Data model for a time slot in the FLL Scheduler GA."""

    start: datetime
    stop: datetime
    start_str: str
    stop_str: str

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        return f"{self.start_str}-{self.stop_str}"

    def __lt__(self, other: "TimeSlot") -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop and other.start < self.stop
