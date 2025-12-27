"""Event data model for FLL scheduling."""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .location import Location
from .timeslot import TimeSlot

if TYPE_CHECKING:
    from ..config.app_schemas import TournamentConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventProperties:
    """Holds properties of an event for fast access during evaluation."""

    all_props: np.ndarray
    roundtype: np.ndarray
    roundtype_idx: np.ndarray
    timeslot: np.ndarray
    timeslot_idx: np.ndarray
    start: np.ndarray
    stop_active: np.ndarray
    stop_cycle: np.ndarray
    location: np.ndarray
    loc_str: np.ndarray
    loc_type: np.ndarray
    loc_idx: np.ndarray
    loc_name: np.ndarray
    loc_side: np.ndarray
    teams_per_round: np.ndarray
    paired_idx: np.ndarray

    @classmethod
    def build(cls, n_total_events: int, event_map: dict[int, Event]) -> EventProperties:
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
        event_properties = np.zeros(n_total_events, dtype=event_prop_dtype)
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
        return cls(
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


@dataclass(slots=True)
class Event:
    """Data model for an event in a schedule."""

    idx: int
    roundtype: str
    roundtype_idx: int
    timeslot: TimeSlot
    location: Location
    paired: int = 0
    conflicts: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        """Get string representation of Event."""
        return f"{self.idx}, {self.roundtype}, {self.location}, {self.timeslot}"

    def pair(self, other: Event) -> None:
        """Pair this event with another event."""
        self.paired = other.idx
        other.paired = self.idx

    @classmethod
    def build_null(cls) -> Event:
        """Build and return a Null Event."""
        return cls(
            idx=0,
            roundtype="n",
            roundtype_idx=0,
            timeslot=TimeSlot.build_null(),
            location=Location.build_null(),
        )


@dataclass(slots=True)
class EventFactory:
    """Factory class to create Events based on TournamentRound configurations."""

    config: TournamentConfig
    _list: tuple[Event, ...] = ()
    _list_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    _list_singles_or_side1: tuple[Event, ...] = ()
    _list_singles_or_side1_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    _conflict_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    _cached_conflict_map: dict[int, set[int]] = field(default_factory=dict)
    _cached_mapping: dict[int, Event] = field(default_factory=dict)
    _cached_roundtypes: dict[int, list[int]] = field(default_factory=dict)
    _cached_timeslots: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    _cached_locations: dict[tuple[str, Location], list[Event]] = field(default_factory=dict)
    _cached_matches: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.build()
        self.build_indices()
        self.build_singles_or_side1()
        self.build_singles_or_side1_indices()
        self.build_conflicts()
        self.build_conflict_matrix()
        self.as_mapping()
        self.as_timeslots()
        self.as_matches()
        self.as_roundtypes()

    def build(self) -> tuple[Event, ...]:
        """Create and return all Events for the tournament."""
        if not self._list:
            event_idx_iter = itertools.count()
            _list = []
            _list.extend(e for r in self.config.rounds for e in r.create_events(event_idx_iter))
            self._list = tuple(_list)
        return self._list

    def build_indices(self) -> np.ndarray:
        """Create and return a list of all event indices."""
        if self._list_indices.size == 0:
            self._list_indices = np.array([e.idx for e in self.build()], dtype=int)
        return self._list_indices

    def build_singles_or_side1(self) -> tuple[Event, ...]:
        """Create and return all single-team Events or side 1 of paired Events."""
        if not self._list_singles_or_side1:
            _list_singles_or_side1 = []
            _list_singles_or_side1.extend(
                e for e in self.build() if e.paired == -1 or (e.paired != -1 and e.location.side == 1)
            )
            self._list_singles_or_side1 = tuple(_list_singles_or_side1)
        return self._list_singles_or_side1

    def build_singles_or_side1_indices(self) -> np.ndarray:
        """Create and return all single-team Events or side 1 of paired Events."""
        if self._list_singles_or_side1_indices.size == 0:
            self._list_singles_or_side1_indices = np.array([e.idx for e in self.build_singles_or_side1()], dtype=int)
        return self._list_singles_or_side1_indices

    def build_conflicts(self) -> None:
        """Build a mapping of event identities to their conflicting events."""
        for e1, e2 in itertools.combinations(self.build(), 2):
            if e1.timeslot.overlaps(e2.timeslot):
                e1.conflicts.append(e2.idx)
                e2.conflicts.append(e1.idx)

        for e in self.build():
            e.conflicts = sorted(set(e.conflicts))
            logger.debug("%s has %d conflicts: %s", e, len(e.conflicts), e.conflicts)

    def build_conflict_matrix(self) -> np.ndarray:
        """Build a conflict matrix for all events."""
        if self._conflict_matrix.size == 0:
            n = len(self._list)
            self._conflict_matrix = np.full((n, n), fill_value=False, dtype=bool)
            for e1, e2 in itertools.combinations(self.build(), 2):
                if e1.timeslot.overlaps(e2.timeslot):
                    self._conflict_matrix[e1.idx, e2.idx] = True
                    self._conflict_matrix[e2.idx, e1.idx] = True
            for i in range(n):
                self._conflict_matrix[i, i] = True  # An event conflicts with itself
        return self._conflict_matrix

    def as_mapping(self) -> dict[int, Event]:
        """Get a mapping of event identities to Event objects."""
        if not self._cached_mapping:
            self._cached_mapping.update((e.idx, e) for e in self.build())
        return self._cached_mapping

    def as_conflict_map(self) -> dict[int, set[int]]:
        """Get a mapping of event identities to their conflicting event indices."""
        if not self._cached_conflict_map:
            self._cached_conflict_map = {e.idx: set(e.conflicts) for e in self.build()}
        return self._cached_conflict_map

    def as_roundtypes(self) -> dict[int, list[int]]:
        """Get a mapping of RoundTypes to their Event indices."""
        if not self._cached_roundtypes:
            self._cached_roundtypes = defaultdict(list)
            for e in self.build():
                self._cached_roundtypes[e.roundtype_idx].append(e.idx)
        return self._cached_roundtypes

    def as_timeslots(self) -> dict[tuple[int, int], list[int]]:
        """Get a mapping of TimeSlots to their Events."""
        if not self._cached_timeslots:
            self._cached_timeslots = defaultdict(list)
            for e in self.build():
                self._cached_timeslots[(e.roundtype_idx, e.timeslot.idx)].append(e.idx)
        return self._cached_timeslots

    def as_matches(self) -> dict[int, list[tuple[int, int]]]:
        """Get a mapping of RoundTypes to their matched Event indices."""
        if not self._cached_matches:
            self._cached_matches = defaultdict(list)
            for e in self.build_singles_or_side1():
                if e.paired == -1:
                    continue
                self._cached_matches[e.roundtype_idx].append((e.idx, e.paired))
        return self._cached_matches
