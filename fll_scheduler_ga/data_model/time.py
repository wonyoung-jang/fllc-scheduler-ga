"""Time data module for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(slots=True, unsafe_hash=True)
class TimeSlot:
    """Data model for a time slot in the FLL Scheduler GA."""

    idx: int

    start: datetime
    stop: datetime

    time_fmt: ClassVar[str]

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.start >= self.stop:
            msg = "Start time must be before stop time."
            raise ValueError(msg)

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        return f"{self.start.strftime(TimeSlot.time_fmt)}-{self.stop.strftime(TimeSlot.time_fmt)}"

    def __lt__(self, other: TimeSlot) -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    @classmethod
    def set_time_format(cls, time_format: str) -> None:
        """Set the time format for the time slot."""
        cls.time_fmt = time_format

    def overlaps(self, other: TimeSlot) -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop and other.start < self.stop
