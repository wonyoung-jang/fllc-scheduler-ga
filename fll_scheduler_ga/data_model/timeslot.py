"""Time data module for FLL Scheduler GA."""

from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel


def parse_time_str(dt_str: str | None, fmt: str) -> datetime | None:
    """Parse a time string into a datetime object."""
    if not dt_str:
        return None
    return datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=UTC)


class TimeSlot(BaseModel):
    """Data model for a time slot in the FLL Scheduler GA."""

    model_config = {"arbitrary_types_allowed": True, "frozen": True}
    idx: int
    start: datetime | None
    stop_active: datetime | None
    stop_cycle: datetime | None
    time_fmt: ClassVar[str]

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        fmt = TimeSlot.time_fmt
        return f"{self.start.strftime(fmt)}-{self.stop_cycle.strftime(fmt)}"

    def __lt__(self, other: "TimeSlot") -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop_cycle and other.start < self.stop_cycle
