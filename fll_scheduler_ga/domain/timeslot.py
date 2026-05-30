"""Time data module for the Tournament Scheduler."""

import datetime as dt
from dataclasses import dataclass
from typing import ClassVar

DEFAULT_DT = dt.datetime.min.replace(tzinfo=dt.UTC)


@dataclass(slots=True, frozen=True)
class TimeSlot:
    """Data model for a time slot in the FLL Scheduler GA."""

    idx: int = 0
    start: dt.datetime = DEFAULT_DT
    stop_active: dt.datetime = DEFAULT_DT
    stop_cycle: dt.datetime = DEFAULT_DT
    time_fmt: ClassVar[str]

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        return f"{self.start.strftime(TimeSlot.time_fmt)}-{self.stop_cycle.strftime(TimeSlot.time_fmt)}"

    def __lt__(self, other: TimeSlot) -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: TimeSlot) -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop_cycle and other.start < self.stop_cycle
