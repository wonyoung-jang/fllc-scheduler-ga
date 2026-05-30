"""Event data model for FLL scheduling."""

import logging
from dataclasses import dataclass, field

import numpy as np

from fll_scheduler_ga.domain.location import Location
from fll_scheduler_ga.domain.timeslot import TimeSlot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventProperties:
    """Holds properties of an event for fast access during evaluation."""

    all_props: np.ndarray = field(default_factory=lambda: np.array([]))
    roundtype: np.ndarray = field(default_factory=lambda: np.array([]))
    roundtype_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    timeslot: np.ndarray = field(default_factory=lambda: np.array([]))
    timeslot_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    start: np.ndarray = field(default_factory=lambda: np.array([]))
    stop_active: np.ndarray = field(default_factory=lambda: np.array([]))
    stop_cycle: np.ndarray = field(default_factory=lambda: np.array([]))
    location: np.ndarray = field(default_factory=lambda: np.array([]))
    loc_str: np.ndarray = field(default_factory=lambda: np.array([]))
    loc_type: np.ndarray = field(default_factory=lambda: np.array([]))
    loc_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    loc_name: np.ndarray = field(default_factory=lambda: np.array([]))
    loc_side: np.ndarray = field(default_factory=lambda: np.array([]))
    teams_per_round: np.ndarray = field(default_factory=lambda: np.array([]))
    paired_idx: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    idx: int = 0
    roundtype: str = "Null"
    roundtype_idx: int = 0
    timeslot: TimeSlot = field(default_factory=TimeSlot)
    location: Location = field(default_factory=Location)
    paired: int = -1
    conflicts: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.idx}, {self.roundtype}, {self.location}, {self.timeslot}"

    def pair(self, other: Event) -> None:
        """Pair this event with another event."""
        self.paired = other.idx
        other.paired = self.idx


def build_event_props(n_total_events: int, event_map: dict[int, Event]) -> EventProperties:
    """Build EventProperties from an event mapping."""
    event_prop_dtype = np.dtype(
        [
            ("roundtype", "U50"),
            ("roundtype_idx", int),
            ("timeslot", object),
            ("timeslot_idx", int),
            ("start", int),
            ("stop_active", int),
            ("stop_cycle", int),
            ("location", object),
            ("loc_str", "U50"),
            ("loc_type", "U50"),
            ("loc_idx", int),
            ("loc_name", int),
            ("loc_side", int),
            ("teams_per_round", int),
            ("paired_idx", int),
        ]
    )
    event_properties: np.ndarray = np.zeros(n_total_events, dtype=event_prop_dtype)
    for i in range(n_total_events):
        e = event_map[i]
        event_properties[i]["roundtype"] = e.roundtype
        event_properties[i]["roundtype_idx"] = e.roundtype_idx
        event_properties[i]["timeslot"] = e.timeslot
        event_properties[i]["timeslot_idx"] = e.timeslot.idx
        event_properties[i]["start"] = int(e.timeslot.start.timestamp())
        event_properties[i]["stop_active"] = int(e.timeslot.stop_active.timestamp())
        event_properties[i]["stop_cycle"] = int(e.timeslot.stop_cycle.timestamp())
        event_properties[i]["location"] = e.location
        event_properties[i]["loc_str"] = str(e.location)
        event_properties[i]["loc_type"] = e.location.locationtype
        event_properties[i]["loc_idx"] = e.location.idx
        event_properties[i]["loc_name"] = e.location.name
        event_properties[i]["loc_side"] = e.location.side
        event_properties[i]["teams_per_round"] = e.location.teams_per_round
        event_properties[i]["paired_idx"] = e.paired
    names = event_properties.dtype.names
    names = names if isinstance(names, tuple) else ("",)
    event_prop_labels = ", ".join(names)
    logger.debug("\nEvent properties array:\n%s\n%s", event_prop_labels, event_properties)
    return EventProperties(
        all_props=event_properties,
        roundtype=event_properties["roundtype"],
        roundtype_idx=event_properties["roundtype_idx"],
        timeslot=event_properties["timeslot"],
        timeslot_idx=event_properties["timeslot_idx"],
        start=event_properties["start"],
        stop_active=event_properties["stop_active"],
        stop_cycle=event_properties["stop_cycle"],
        location=event_properties["location"],
        loc_str=event_properties["loc_str"],
        loc_type=event_properties["loc_type"],
        loc_idx=event_properties["loc_idx"],
        loc_name=event_properties["loc_name"],
        loc_side=event_properties["loc_side"],
        teams_per_round=event_properties["teams_per_round"],
        paired_idx=event_properties["paired_idx"],
    )
