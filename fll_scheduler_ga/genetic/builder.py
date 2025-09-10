"""Builder for creating a valid schedule individual."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..data_model.schedule import Schedule

if TYPE_CHECKING:
    from random import Random

    from ..config.config import RoundType, TournamentConfig
    from ..data_model.event import Event, EventFactory
    from ..data_model.team import Team, TeamFactory

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    rng: Random

    def build(self, rng: Random | None = None) -> Schedule:
        """Construct and return the final schedule."""
        rng = self.rng if rng is None else rng
        events = {}
        for rt, evts in self.event_factory.as_roundtypes().items():
            events[rt] = rng.sample(evts, k=len(evts))

        schedule = Schedule(teams=self.team_factory.build(), origin="Builder")
        teams = rng.sample(schedule.all_teams(), k=self.config.num_teams)

        build_map = {
            1: self.build_singles,
            2: self.build_matches,
        }
        tpr_reqs = {rc.roundtype: rc.teams_per_round for rc in self.config.rounds}
        for rt, evts in events.items():
            build_fn = build_map.get(tpr_reqs.get(rt))
            if build_fn:
                build_fn(schedule, rt, evts, teams)

        schedule.clear_cache()
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
