"""Builder for creating a valid schedule individual."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.tournament_config import TournamentConfig

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    event_factory: EventFactory
    event_properties: EventProperties
    config: TournamentConfig
    rng: np.random.Generator
    roundtype_events: dict[int, list[int]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the random number generator."""
        self.roundtype_events = self.event_factory.as_roundtypes()

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        schedule = Schedule(origin="Builder")
        for roundtype, evts in self.roundtype_events.items():
            events = self.rng.permutation(evts)
            if self.config.round_idx_to_tpr[roundtype] == 1:
                self.build_singles(schedule, events, roundtype)
            elif self.config.round_idx_to_tpr[roundtype] == 2:
                self.build_matches(schedule, events, roundtype)
        return schedule

    def build_singles(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            available = (
                t
                for t in self.rng.permutation(schedule.all_rounds_needed(roundtype))
                if not schedule.conflicts(t, event)
            )
            team = next(available, None)
            if team:
                schedule.assign(team, event)

    def build_matches(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all events for a specific round type."""
        side1s = events[np.nonzero(self.event_properties.loc_side[events] == 1)[0]]
        side2s = self.event_properties.paired_idx[side1s]
        for e1, e2 in zip(side1s, side2s, strict=True):
            available = (
                t
                for t in self.rng.permutation(schedule.all_rounds_needed(roundtype))
                if not schedule.conflicts(t, e1) and not schedule.conflicts(t, e2)
            )
            t1 = next(available, None)
            t2 = next(available, None)
            if t1 and t2:
                schedule.assign(t1, e1)
                schedule.assign(t2, e2)


@dataclass(slots=True)
class ScheduleBuilderNaive:
    """Naive builder for building a valid minimally random schedule."""

    event_factory: EventFactory
    event_properties: EventProperties
    config: TournamentConfig
    rng: np.random.Generator
    roundtype_events: dict[int, list[int]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the random number generator."""
        self.roundtype_events = self.event_factory.as_roundtypes()

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        schedule = Schedule(origin="Naive Builder")
        for _ in range(3):
            for roundtype, evts in self.roundtype_events.items():
                events = np.array(evts, dtype=int)
                if self.config.round_idx_to_tpr[roundtype] == 1:
                    self.build_singles(schedule, events, roundtype)
                elif self.config.round_idx_to_tpr[roundtype] == 2:
                    self.build_matches(schedule, events, roundtype)
        return schedule

    def build_singles(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            available = (t for t in schedule.all_rounds_needed(roundtype) if not schedule.conflicts(t, event))
            team = next(available, -1)
            if team != -1:
                schedule.assign(team, event)

    def build_matches(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all events for a specific round type."""
        side1s = events[np.nonzero(self.event_properties.loc_side[events] == 1)[0]]
        side2s = self.event_properties.paired_idx[side1s]
        for side1, side2 in zip(side1s, side2s, strict=True):
            available = (
                t
                for t in schedule.all_rounds_needed(roundtype)
                if not schedule.conflicts(t, side1) and not schedule.conflicts(t, side2)
            )
            team1 = next(available, -1)
            team2 = next(available, -1)
            if team1 != -1 and team2 != -1:
                schedule.assign(team1, side1)
                schedule.assign(team2, side2)
