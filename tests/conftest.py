"""Fixtures for testing fll_scheduler_ga package."""

from datetime import UTC, datetime

import pytest

from fll_scheduler_ga.data_model.timeslot import TimeSlot

FMT_24H = "%H:%M"
FMT_12H = "%I:%M %p"


@pytest.fixture
def timeslot() -> TimeSlot:
    """Create a sample TimeSlot for testing."""
    start = datetime.strptime("09:00", FMT_24H).replace(tzinfo=UTC)
    stop_active = datetime.strptime("09:15", FMT_24H).replace(tzinfo=UTC)
    stop_cycle = datetime.strptime("10:00", FMT_24H).replace(tzinfo=UTC)
    TimeSlot.time_fmt = FMT_24H
    return TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
