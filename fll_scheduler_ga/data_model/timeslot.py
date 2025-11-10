"""Time data module for FLL Scheduler GA."""

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel


class TimeSlot(BaseModel):
    """Data model for a time slot in the FLL Scheduler GA."""

    model_config = {"arbitrary_types_allowed": True, "frozen": True}
    idx: int
    start: datetime
    stop: datetime
    time_fmt: ClassVar[str]

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        fmt = TimeSlot.time_fmt
        return f"{self.start.strftime(fmt)}-{self.stop.strftime(fmt)}"

    def __lt__(self, other: "TimeSlot") -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: "TimeSlot") -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop and other.start < self.stop


TimeSlot.model_rebuild()
