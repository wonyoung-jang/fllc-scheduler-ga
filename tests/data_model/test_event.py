"""Tests for event."""

from typing import TYPE_CHECKING

from fll_scheduler_ga.domain.model import Event

if TYPE_CHECKING:
    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TimeSlot


def test_event_str(timeslot: TimeSlot) -> None:
    """Test the string representation of Event."""
    event = Event(timeslot=timeslot)
    assert str(event) == f"0, Null, {event.location!s}, {event.timeslot!s}"


def test_event_factory_and_properties(evt_repo: EventRepository, evt_prop: EventProperties) -> None:
    """Test EventRepository and EventProperties."""
    events = evt_repo.events
    assert len(events) > 0
    assert evt_repo.events_idx.size == len(events)
    # Check conflicts
    conf_map = evt_repo.conflict_map
    assert isinstance(conf_map, dict)
    # Properties
    ep = evt_prop
    assert ep.timeslot_idx[0] == events[0].timeslot.idx
