"""Tests for time-related functionality."""

from datetime import UTC, datetime, timedelta

import pytest

from fll_scheduler_ga.domain.model import DEFAULT_DT, TimeSlot
from fll_scheduler_ga.domain.schema import (
    TIME_FORMAT_MAP,
    _calc_num_timeslots,
    _infer_time_format,
    _init_timeslots,
    _parse_time_str,
    _validate_duration,
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
    """Parametrized tests for _parse_time_str function."""
    result = _parse_time_str(dt_str, fmt)
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
    start = _parse_time_str(start_str, fmt)
    stop_active = _parse_time_str(stop_active_str, fmt)
    stop_cycle = _parse_time_str(stop_cycle_str, fmt)
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
    ids=["enough_times", "enough_locations", "cannot_calculate"],
)
def test_calc_num_timeslots(
    n_times: int, n_locs: int, n_teams: int, rounds_per_team: int, expected: int | None
) -> None:
    """Parametrized tests for _calc_num_timeslots function."""
    if expected is not None:
        result = _calc_num_timeslots(n_times, n_locs, n_teams, rounds_per_team)
        assert result == expected
    else:
        with pytest.raises(ValueError, match=r"Cannot calculate number of timeslots without times or locations."):
            _calc_num_timeslots(n_times, n_locs, n_teams, rounds_per_team)


@pytest.mark.parametrize(
    ("dt_str", "expected"),
    [
        ("4:00 AM", FMT_12H),
        ("4:00 PM", FMT_12H),
        ("06:00", FMT_24H),
        ("16:00", FMT_24H),
        ("09:30:45", None),
        ("9PM", None),
    ],
    ids=["12h_am", "12h_pm", "24h_morning", "24h_afternoon", "with_seconds", "no_minutes"],
)
def test_infer_time_format(dt_str: str, expected: str) -> None:
    """Test inferring time format from string."""
    inferred = _infer_time_format(dt_str)
    assert inferred == expected or (inferred is None and expected is None)


class TestValidateDuration:
    """Tests for _validate_duration function."""

    @pytest.mark.parametrize(
        ("start_dt", "dur", "expected_minutes"),
        [
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), 30, 30),
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), 15, 15),
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), 60, 60),
        ],
        ids=["30min", "15min", "60min"],
    )
    def test_validate_duration_with_start_and_duration(
        self, start_dt: datetime, dur: int, expected_minutes: int
    ) -> None:
        """Test _validate_duration with start time and duration specified."""
        result = _validate_duration(start_stop=(start_dt, DEFAULT_DT), times_dt=(), dur=dur, n_timeslots=0)
        assert result == timedelta(minutes=expected_minutes)

    @pytest.mark.parametrize(
        ("times", "dur", "expected_minutes"),
        [
            ((datetime(2026, 1, 1, 9, 0, tzinfo=UTC),), 20, 20),
            ((datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 10, 0, tzinfo=UTC)), 45, 45),
        ],
        ids=["single_time", "multiple_times"],
    )
    def test_validate_duration_with_times_and_duration(
        self, times: tuple[datetime, ...], dur: int, expected_minutes: int
    ) -> None:
        """Test _validate_duration with explicit times and duration specified."""
        result = _validate_duration(start_stop=(DEFAULT_DT, DEFAULT_DT), times_dt=times, dur=dur, n_timeslots=0)
        assert result == timedelta(minutes=expected_minutes)

    @pytest.mark.parametrize(
        ("start_dt", "stop_dt", "n_timeslots", "expected_minutes"),
        [
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 10, 0, tzinfo=UTC), 4, 15),
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 11, 0, tzinfo=UTC), 6, 20),
            (datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 10, 30, tzinfo=UTC), 3, 30),
        ],
        ids=["60min_4slots", "120min_6slots", "90min_3slots"],
    )
    def test_validate_duration_with_start_stop_calculates(
        self, start_dt: datetime, stop_dt: datetime, n_timeslots: int, expected_minutes: int
    ) -> None:
        """Test _validate_duration calculates duration from start/stop times."""
        result = _validate_duration(start_stop=(start_dt, stop_dt), times_dt=(), dur=0, n_timeslots=n_timeslots)
        assert result == timedelta(minutes=expected_minutes)

    def test_validate_duration_start_stop_invalid_n_timeslots(self) -> None:
        """Test _validate_duration raises error when n_timeslots is invalid."""
        with pytest.raises(ValueError, match=r"n_timeslots must be greater than zero"):
            _validate_duration(
                start_stop=(datetime(2026, 1, 1, 9, 0, tzinfo=UTC), datetime(2026, 1, 1, 10, 0, tzinfo=UTC)),
                times_dt=(),
                dur=0,
                n_timeslots=0,
            )

    def test_validate_duration_with_default_dt_and_no_dur_raises_error(self) -> None:
        """Test _validate_duration with DEFAULT_DT values and no duration raises error."""
        with pytest.raises(ValueError, match=r"n_timeslots must be greater than zero"):
            _validate_duration(start_stop=(DEFAULT_DT, DEFAULT_DT), times_dt=(), dur=0, n_timeslots=0)

    def test_validate_duration_minimum_one_minute(self) -> None:
        """Test _validate_duration returns minimum 1 minute for very small durations."""
        result = _validate_duration(
            start_stop=(
                datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 0, 30, tzinfo=UTC),  # 30 seconds
            ),
            times_dt=(),
            dur=0,
            n_timeslots=1,
        )
        assert result == timedelta(minutes=1)


class TestInitTimeslots:
    """Tests for _init_timeslots function."""

    def test_init_timeslots_with_explicit_starts(self) -> None:
        """Test _init_timeslots with explicit start times provided."""
        starts = (
            datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        )
        dur_cycle = timedelta(minutes=30)
        dur_active = timedelta(minutes=20)
        result = tuple(_init_timeslots(starts, dur_cycle, dur_active, 0, DEFAULT_DT))
        assert len(result) == 3
        # Check first timeslot
        assert result[0][0] == datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        assert result[0][1] == datetime(2026, 1, 1, 9, 20, tzinfo=UTC)  # start + active
        assert result[0][2] == datetime(2026, 1, 1, 9, 30, tzinfo=UTC)  # next start
        # Check last timeslot
        assert result[2][0] == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        assert result[2][1] == datetime(2026, 1, 1, 10, 20, tzinfo=UTC)
        assert result[2][2] == datetime(2026, 1, 1, 10, 30, tzinfo=UTC)  # last + cycle

    def test_init_timeslots_with_start_dt_and_count(self) -> None:
        """Test _init_timeslots generating slots from start datetime and count."""
        start_dt = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        dur_cycle = timedelta(minutes=30)
        dur_active = timedelta(minutes=20)
        n_timeslots = 4
        result = tuple(_init_timeslots((), dur_cycle, dur_active, n_timeslots, start_dt))
        assert len(result) == 4
        # Check first timeslot
        assert result[0][0] == datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        assert result[0][1] == datetime(2026, 1, 1, 9, 20, tzinfo=UTC)
        assert result[0][2] == datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
        # Check second timeslot
        assert result[1][0] == datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
        assert result[1][1] == datetime(2026, 1, 1, 9, 50, tzinfo=UTC)
        assert result[1][2] == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        # Check last timeslot
        assert result[3][0] == datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
        assert result[3][1] == datetime(2026, 1, 1, 10, 50, tzinfo=UTC)
        assert result[3][2] == datetime(2026, 1, 1, 11, 0, tzinfo=UTC)

    def test_init_timeslots_single_slot(self) -> None:
        """Test _init_timeslots with single timeslot generation."""
        start_dt = datetime(2026, 1, 1, 14, 0, tzinfo=UTC)
        dur_cycle = timedelta(minutes=45)
        dur_active = timedelta(minutes=30)
        result = tuple(_init_timeslots((), dur_cycle, dur_active, 1, start_dt))
        assert len(result) == 1
        assert result[0][0] == datetime(2026, 1, 1, 14, 0, tzinfo=UTC)
        assert result[0][1] == datetime(2026, 1, 1, 14, 30, tzinfo=UTC)
        assert result[0][2] == datetime(2026, 1, 1, 14, 45, tzinfo=UTC)

    def test_init_timeslots_empty_when_no_timeslots(self) -> None:
        """Test _init_timeslots returns empty iterator when n_timeslots is 0."""
        start_dt = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        dur_cycle = timedelta(minutes=30)
        dur_active = timedelta(minutes=20)
        result = tuple(_init_timeslots((), dur_cycle, dur_active, 0, start_dt))
        assert len(result) == 0

    def test_init_timeslots_with_different_active_cycle_durations(self) -> None:
        """Test _init_timeslots where active duration differs from cycle."""
        start_dt = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        dur_cycle = timedelta(minutes=60)
        dur_active = timedelta(minutes=15)
        n_timeslots = 2
        result = tuple(_init_timeslots((), dur_cycle, dur_active, n_timeslots, start_dt))
        assert len(result) == 2
        # First slot: 9:00-9:15 (active), 9:00-10:00 (cycle)
        assert result[0][0] == datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        assert result[0][1] == datetime(2026, 1, 1, 9, 15, tzinfo=UTC)
        assert result[0][2] == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        # Second slot: 10:00-10:15 (active), 10:00-11:00 (cycle)
        assert result[1][0] == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        assert result[1][1] == datetime(2026, 1, 1, 10, 15, tzinfo=UTC)
        assert result[1][2] == datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
