"""Tests for event."""

from fll_scheduler_ga.data_model.event import Event, EventFactory, EventProperties
from fll_scheduler_ga.data_model.timeslot import TimeSlot


def test_event_str(timeslot: TimeSlot) -> None:
    """Test the string representation of Event."""
    event = Event(timeslot=timeslot)
    assert str(event) == f"0, Null, {event.location!s}, {event.timeslot!s}"


def test_event_factory_and_properties(event_factory: EventFactory, event_properties: EventProperties) -> None:
    """Test EventFactory and EventProperties."""
    events = event_factory.build()
    assert len(events) > 0
    assert event_factory.build_indices().size == len(events)

    # Check conflicts
    conf_map = event_factory.as_conflict_map()
    assert isinstance(conf_map, dict)

    # Properties
    ep = event_properties
    assert ep.all_props.size == len(events)
    assert ep.timeslot_idx[0] == events[0].timeslot.idx
