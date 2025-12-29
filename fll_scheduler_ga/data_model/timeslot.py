"""Time data module for the Tournament Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_DT = datetime.min.replace(tzinfo=UTC)
TIME_FORMAT_MAP = {
    12: "%I:%M %p",
    24: "%H:%M",
}


@dataclass(slots=True, frozen=True)
class TimeSlot:
    """Data model for a time slot in the FLL Scheduler GA."""

    idx: int = 0
    start: datetime = DEFAULT_DT
    stop_active: datetime = DEFAULT_DT
    stop_cycle: datetime = DEFAULT_DT

    time_fmt: ClassVar[str]

    def __str__(self) -> str:
        """Get a string representation of the time slot."""
        fmt = TimeSlot.time_fmt
        return f"{self.start.strftime(fmt)}-{self.stop_cycle.strftime(fmt)}"

    def __lt__(self, other: TimeSlot) -> bool:
        """Less-than comparison based on start time."""
        return self.start < other.start

    def overlaps(self, other: TimeSlot) -> bool:
        """Check if this time slot overlaps with another."""
        return self.start < other.stop_cycle and other.start < self.stop_cycle


def parse_time_str(dt_str: str, fmt: str) -> datetime:
    """Parse a time string into a datetime object."""
    if not dt_str:
        return DEFAULT_DT
    return datetime.strptime(dt_str.strip(), fmt).replace(tzinfo=UTC)


def infer_time_format(dt_str: str) -> str | None:
    """Infer the time format from a sample time string."""
    for fmt in TIME_FORMAT_MAP.values():
        try:
            parse_time_str(dt_str, fmt)
        except ValueError:
            continue
        return fmt
    return None


def calc_num_timeslots(n_times: int, n_locs: int, n_teams: int, rounds_per_team: int) -> int:
    """Calculate the number of timeslots needed for a round."""
    if n_times > 0:
        return n_times

    if n_locs > 0:
        return ceil((n_teams * rounds_per_team) / n_locs)

    msg = "Cannot calculate number of timeslots without times or locations."
    raise ValueError(msg)


def validate_duration(
    start_dt: datetime,
    stop_dt: datetime,
    times_dt: tuple[datetime, ...],
    dur: int,
    n_timeslots: int,
) -> timedelta:
    """Validate the times configuration for a round.

    Valid conditions:
    1. start + duration
    2. times + duration
    3. start + stop (need to calculate num_timeslots)
    """
    if (start_dt != DEFAULT_DT or times_dt) and dur:
        return timedelta(minutes=dur)

    if start_dt and stop_dt:
        if n_timeslots <= 0:
            msg = "n_timeslots must be greater than zero to validate duration."
            raise ValueError(msg)
        diff = stop_dt - start_dt
        total_available = diff.total_seconds()
        minimum_duration = total_available // n_timeslots
        return timedelta(minutes=max(1, minimum_duration // 60))

    return timedelta(minutes=0)


def init_timeslots(
    starts: tuple[datetime, ...],
    dur_cycle: timedelta,
    dur_active: timedelta,
    n_timeslots: int,
    start_dt: datetime,
) -> Iterator[tuple[datetime, ...]]:
    """Initialize the timeslots for the round."""
    if starts and dur_active and dur_cycle:
        stops_cycle = (*starts[1:], starts[-1] + dur_cycle)
        stops_active = (start + dur_active for start in starts)
        yield from zip(starts, stops_active, stops_cycle, strict=True)
        return

    current = start_dt
    for _ in range(n_timeslots):
        stop_cycle = current + dur_cycle
        stop_active = current + dur_active
        yield (current, stop_active, stop_cycle)
        current = stop_cycle
