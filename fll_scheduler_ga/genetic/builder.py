"""Builder for creating a valid schedule individual."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    import numpy as np

    from ..data_model.config import RoundType, TournamentConfig
    from ..data_model.event import Event, EventFactory
    from ..data_model.team import Team, TeamFactory

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    rng: np.random.Generator
    roundtype_events: dict[RoundType, list[Event]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the random number generator."""
        self.roundtype_events = self.event_factory.as_roundtypes()

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        events = {}
        for rt, evts in self.roundtype_events.items():
            events[rt] = self.rng.permutation(evts)

        schedule = Schedule(
            teams=self.team_factory.build(),
            origin="Builder",
        )
        teams = self.rng.permutation(schedule.teams)

        for rt, evts in events.items():
            if self.config.round_to_tpr[rt] == 1:
                self.build_singles(schedule, rt, evts, teams)
            elif self.config.round_to_tpr[rt] == 2:
                self.build_matches(schedule, rt, evts, teams)
        return schedule

    def build_singles(self, schedule: Schedule, rt: RoundType, events: list[Event], teams: list[Team]) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            available = (t for t in teams if t.needs_round(rt) and not t.conflicts(event))
            team = next(available, None)
            if team:
                schedule.assign_single(event, team)

    def build_matches(self, schedule: Schedule, rt: RoundType, events: list[Event], teams: list[Team]) -> None:
        """Book all events for a specific round type."""
        for side1, side2 in ((e, e.paired) for e in events if e.location.side == 1):
            available = (t for t in teams if t.needs_round(rt) and not t.conflicts(side1))
            team1 = next(available, None)
            team2 = next(available, None)
            if team1 and team2:
                schedule.assign_match(side1, side2, team1, team2)
