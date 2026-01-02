"""Tests for schedule."""

from fll_scheduler_ga.data_model.event import EventFactory
from fll_scheduler_ga.data_model.schedule import Schedule


def test_schedule_operations(empty_schedule: Schedule, event_factory: EventFactory) -> None:
    """Test Schedule assignment logic."""
    s = empty_schedule
    events = event_factory.build()
    e_idx = events[0].idx

    # Assign
    s.assign(0, e_idx)
    assert s.schedule[e_idx] == 0
    assert s.get_size() == 1
    assert e_idx in s.schedule

    # Unassign
    s.unassign(0, e_idx)
    assert s.schedule[e_idx] == -1
    assert s.get_size() == 0

    # Swap
    s.assign(0, e_idx)
    e_idx_2 = events[1].idx
    s.swap_assignment(0, e_idx, e_idx_2)
    assert s.schedule[e_idx] == -1
    assert s.schedule[e_idx_2] == 0

    # Assign null team
    s.assign(-1, e_idx)
    assert s.schedule[e_idx] == -1

    # Unassign null team
    s.unassign(-1, e_idx_2)
    assert s.schedule[e_idx_2] == 0

    # Swap null team
    s.swap_assignment(-1, e_idx_2, e_idx)
    assert s.schedule[e_idx_2] == 0
    assert s.schedule[e_idx] == -1

    # Conflict checks
    assert s.conflicts(0, e_idx_2)  # with assigned event
    assert not s.conflicts(-1, e_idx)  # null team
    assert not s.conflicts(0, e_idx_2, ignore=e_idx_2)  # ignoring assigned event


def test_schedule_clone_and_hash(empty_schedule: Schedule, event_factory: EventFactory) -> None:
    """Test cloning and hashing."""
    s = empty_schedule
    events = event_factory.build()
    s.assign(0, events[0].idx)

    s_clone = s.clone()
    assert s == s_clone

    s.assign(1, events[1].idx)
    assert s != s_clone


def test_schedule_rounds_needed(empty_schedule: Schedule) -> None:
    """Test rounds needed logic."""
    teams = empty_schedule.all_rounds_needed(0)
    assert empty_schedule.any_rounds_needed()
    assert empty_schedule.needs_round(0, 0)
    assert 0 in teams


def test_scheduled_events(empty_schedule: Schedule) -> None:
    """Test retrieval of scheduled events."""
    scheduled = empty_schedule.scheduled_events()
    unscheduled = empty_schedule.unscheduled_events()
    assert scheduled.size == 0
    assert unscheduled.size == empty_schedule.schedule.size
