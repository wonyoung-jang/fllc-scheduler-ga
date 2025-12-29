"""Tests for time-related functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from fll_scheduler_ga.data_model.timeslot import (
    TIME_FORMAT_MAP,
    TimeSlot,
    calc_num_timeslots,
    parse_time_str,
)

FMT_24H = TIME_FORMAT_MAP[24]
FMT_12H = TIME_FORMAT_MAP[12]


@pytest.mark.parametrize(
    ("dt_str", "fmt", "expected"),
    [
        ("09:00 AM", FMT_12H, datetime(1900, 1, 1, 9, 0, tzinfo=UTC)),
        ("09:00", FMT_24H, datetime(1900, 1, 1, 9, 0, tzinfo=UTC)),
        ("", FMT_12H, datetime.min.replace(tzinfo=UTC)),
    ],
    ids=["12h", "24h", "empty"],
)
def test_parse_time_str(dt_str: str, fmt: str, expected: datetime) -> None:
    """Parametrized tests for parse_time_str function."""
    result = parse_time_str(dt_str, fmt)
    assert result == expected


@pytest.mark.parametrize(
    ("start_str", "stop_active_str", "stop_cycle_str", "fmt", "expected"),
    [
        ("09:00 AM", "09:15 AM", "10:00 AM", FMT_12H, "09:00 AM-10:00 AM"),
        ("09:00", "09:15", "10:00", FMT_24H, "09:00-10:00"),
    ],
    ids=["12h", "24h"],
)
def test_timeslot_str(start_str: str, stop_active_str: str, stop_cycle_str: str, fmt: str, expected: str) -> None:
    """Parametrized tests for string representation of TimeSlot."""
    start = parse_time_str(start_str, fmt)
    stop_active = parse_time_str(stop_active_str, fmt)
    stop_cycle = parse_time_str(stop_cycle_str, fmt)
    TimeSlot.time_fmt = fmt
    timeslot = TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
    assert str(timeslot) == expected


@pytest.mark.parametrize(
    ("delta_minutes", "expect_lt", "expect_gt"),
    [
        (-1, True, False),  # earlier
        (1, False, True),  # later
        (0, False, False),  # equal
    ],
    ids=["earlier", "later", "equal"],
)
def test_less_than_timeslot(timeslot: TimeSlot, delta_minutes: int, *, expect_lt: bool, expect_gt: bool) -> None:
    """Parametrized less-than / greater-than comparisons for TimeSlot."""
    other = TimeSlot(
        idx=0,
        start=timeslot.start + timedelta(minutes=delta_minutes),
        stop_active=timeslot.stop_active + timedelta(minutes=delta_minutes),
        stop_cycle=timeslot.stop_cycle + timedelta(minutes=delta_minutes),
    )
    assert (other < timeslot) is expect_lt
    assert (other > timeslot) is expect_gt


@pytest.mark.parametrize(
    ("start_offset_min", "stop_offset_min", "expected"),
    [
        (0, 0, True),  # identical
        (0, -1, True),  # shorter end
        (-1, 0, True),  # earlier start
        (0, 1, True),  # longer end
        (1, 0, True),  # shifted start inside
        (-1, -1, True),  # both earlier but overlapping
        (1, 1, True),  # both later but overlapping
        (-1, 61, True),  # spans across
        (1, 59, True),  # completely inside
        (120, 180, False),  # well after
        (-120, -60, False),  # well before
    ],
    ids=[
        "identical",
        "stop_minus_1",
        "start_minus_1",
        "stop_plus_1",
        "start_plus_1",
        "both_minus_1",
        "both_plus_1",
        "span_across",
        "inside",
        "after_by_2h",
        "before_by_2h",
    ],
)
def test_overlaps_timeslot(timeslot: TimeSlot, start_offset_min: int, stop_offset_min: int, *, expected: bool) -> None:
    """Parametrized overlaps tests.

    Offsets are minutes relative to timeslot.start (so original stop is +60).
    """
    other = TimeSlot(
        idx=0,
        start=timeslot.start + timedelta(minutes=start_offset_min),
        stop_active=timeslot.stop_active + timedelta(minutes=stop_offset_min),
        stop_cycle=timeslot.stop_cycle + timedelta(minutes=stop_offset_min),
    )
    assert other.overlaps(timeslot) is expected


@pytest.mark.parametrize(
    ("n_times", "n_locs", "n_teams", "rounds_per_team", "expected"),
    [
        (5, 0, 10, 2, 5),  # enough times
        (0, 4, 10, 2, 5),  # enough locations
        (0, 0, 10, 2, None),  # cannot calculate
    ],
    ids=[
        "enough_times",
        "enough_locations",
        "cannot_calculate",
    ],
)
def test_calc_num_timeslots(
    n_times: int,
    n_locs: int,
    n_teams: int,
    rounds_per_team: int,
    expected: int | None,
) -> None:
    """Parametrized tests for calc_num_timeslots function."""
    if expected is not None:
        result = calc_num_timeslots(n_times, n_locs, n_teams, rounds_per_team)
        assert result == expected
    else:
        with pytest.raises(ValueError, match=r"Cannot calculate number of timeslots without times or locations."):
            calc_num_timeslots(n_times, n_locs, n_teams, rounds_per_team)
