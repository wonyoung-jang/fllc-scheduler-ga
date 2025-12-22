"""Builder for creating a valid schedule individual."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    import numpy as np

    from ..data_model.event import EventProperties

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder(ABC):
    """Base class for schedule builders."""

    event_properties: EventProperties
    rng: np.random.Generator
    round_idx_to_tpr: dict[int, int]
    roundtype_events: dict[int, list[int]]

    @abstractmethod
    def build(self) -> Schedule:
        """Construct and return the final schedule."""


@dataclass(slots=True)
class ScheduleBuilderRandom(ScheduleBuilder):
    """Builder for building a valid random schedule."""

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        schedule = Schedule(origin="Builder")
        for roundtype, evts in self.roundtype_events.items():
            events = self.rng.permutation(evts)
            if self.round_idx_to_tpr[roundtype] == 1:
                self.build_singles(schedule, events, roundtype)
            elif self.round_idx_to_tpr[roundtype] == 2:
                self.build_matches(schedule, events, roundtype)
        return schedule

    def build_singles(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            all_teams_needing_round = schedule.all_rounds_needed(roundtype)
            shuffled_teams = self.rng.permutation(all_teams_needing_round)
            available = (t for t in shuffled_teams if not schedule.conflicts(t, event))

            if (team := next(available, None)) is not None:
                schedule.assign(team, event)

    def build_matches(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all events for a specific round type."""
        loc_sides = self.event_properties.loc_side[events]
        loc_sides_where_1 = loc_sides == 1
        loc_sides_where_1_idx = loc_sides_where_1.nonzero()[0]

        side1s = events[loc_sides_where_1_idx]
        side2s = self.event_properties.paired_idx[side1s]

        for e1, e2 in zip(side1s, side2s, strict=True):
            all_teams_needing_round = schedule.all_rounds_needed(roundtype)
            shuffled_teams = self.rng.permutation(all_teams_needing_round)
            available = (t for t in shuffled_teams if not schedule.conflicts(t, e1))

            if (t1 := next(available, None)) is not None:
                schedule.assign(t1, e1)

            if (t2 := next(available, None)) is not None:
                schedule.assign(t2, e2)
