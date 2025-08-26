"""Tests for time-related functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from fll_scheduler_ga.data_model.time import TimeSlot

FMT_24H = "%H:%M"
FMT_12H = "%I:%M %p"


@pytest.fixture
def timeslot() -> TimeSlot:
    """Create a sample TimeSlot for testing."""
    start = datetime.strptime("09:00", FMT_24H).replace(tzinfo=UTC)
    stop = datetime.strptime("10:00", FMT_24H).replace(tzinfo=UTC)
    return TimeSlot(start, stop, FMT_24H)


def test_timeslot_str_12h() -> None:
    """Test string representation of TimeSlot."""
    start = datetime.strptime("09:00 AM", FMT_12H).replace(tzinfo=UTC)
    stop = datetime.strptime("10:00 AM", FMT_12H).replace(tzinfo=UTC)
    timeslot = TimeSlot(start, stop, FMT_12H)
    assert str(timeslot) == "09:00 AM-10:00 AM"


def test_timeslot_str_24h() -> None:
    """Test string representation of TimeSlot."""
    start = datetime.strptime("09:00", FMT_24H).replace(tzinfo=UTC)
    stop = datetime.strptime("10:00", FMT_24H).replace(tzinfo=UTC)
    timeslot = TimeSlot(start, stop, FMT_24H)
    assert str(timeslot) == "09:00-10:00"


def test_less_than_timeslot(timeslot: TimeSlot) -> None:
    """Test less than comparison of TimeSlot."""
    earlier = TimeSlot(
        start=timeslot.start - timedelta(hours=1),
        stop=timeslot.stop - timedelta(hours=1),
        time_fmt=timeslot.time_fmt,
    )
    assert earlier < timeslot

    later = TimeSlot(
        start=timeslot.start + timedelta(hours=1),
        stop=timeslot.stop + timedelta(hours=1),
        time_fmt=timeslot.time_fmt,
    )
    assert later > timeslot


def test_overlaps_timeslot(timeslot: TimeSlot) -> None:
    """Test overlaps comparison of TimeSlot."""
    overlapping = TimeSlot(
        start=timeslot.start - timedelta(minutes=30),
        stop=timeslot.stop + timedelta(minutes=30),
        time_fmt=timeslot.time_fmt,
    )
    assert overlapping.overlaps(timeslot)

    non_overlapping = TimeSlot(
        start=timeslot.start + timedelta(hours=1),
        stop=timeslot.stop + timedelta(hours=1),
        time_fmt=timeslot.time_fmt,
    )
    assert not non_overlapping.overlaps(timeslot)
