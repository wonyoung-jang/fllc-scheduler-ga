"""Fixtures for testing fll_scheduler_ga package."""

import pytest

from fll_scheduler_ga.data_model.timeslot import TIME_FORMAT_MAP, TimeSlot, parse_time_str

FMT_24H = TIME_FORMAT_MAP[24]
FMT_12H = TIME_FORMAT_MAP[12]


@pytest.fixture
def timeslot() -> TimeSlot:
    """Create a sample TimeSlot for testing."""
    start = parse_time_str("09:00", FMT_24H)
    stop_active = parse_time_str("09:15", FMT_24H)
    stop_cycle = parse_time_str("10:00", FMT_24H)
    TimeSlot.time_fmt = FMT_24H
    return TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
