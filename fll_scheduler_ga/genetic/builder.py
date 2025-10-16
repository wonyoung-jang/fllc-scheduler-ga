"""Builder for creating a valid schedule individual."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    import numpy as np

    from ..data_model.config import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.team import TeamFactory

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    event_properties: EventProperties
    config: TournamentConfig
    rng: np.random.Generator
    roundtype_events: dict[int, list[int]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the random number generator."""
        self.roundtype_events = self.event_factory.as_roundtype_indices()

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        events = {}
        for rt, evts in self.roundtype_events.items():
            events[rt] = self.rng.permutation(evts)

        schedule = Schedule(
            teams=self.team_factory.teams,
            origin="Builder",
        )
        for rt, evts in events.items():
            if self.config.round_idx_to_tpr[rt] == 1:
                self.build_singles(schedule, evts, rt)
            elif self.config.round_idx_to_tpr[rt] == 2:
                self.build_matches(schedule, evts, rt)
        return schedule

    def build_singles(self, schedule: Schedule, events: list[int], roundtype: int) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            needs_rounds = schedule.all_rounds_needed(roundtype)
            self.rng.shuffle(needs_rounds)
            available = (t for t in needs_rounds if not schedule.conflicts(t, event))
            team = next(available, None)
            if team:
                schedule.assign(team, event)

    def build_matches(self, schedule: Schedule, events: list[int], roundtype: int) -> None:
        """Book all events for a specific round type."""
        for e1 in events:
            e2 = self.event_properties.paired_idx[e1]
            if e2 == -1 or self.event_properties.loc_side[e1] != 1:
                continue
            needs_rounds = schedule.all_rounds_needed(roundtype)
            self.rng.shuffle(needs_rounds)
            available = (t for t in needs_rounds if not schedule.conflicts(t, e1) and not schedule.conflicts(t, e2))
            t1 = next(available, None)
            t2 = next(available, None)
            if t1 and t2:
                schedule.assign(t1, e1)
                schedule.assign(t2, e2)
