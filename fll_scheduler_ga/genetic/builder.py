"""Builder for creating a valid schedule individual."""

import logging
import random
from dataclasses import dataclass, field

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import Team, TeamFactory
from .schedule import Schedule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Encapsulates the logic for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    rng: random.Random
    teams: list[Team] = field(init=False, repr=False)

    def build(self) -> Schedule | None:
        """Construct and return the final schedule."""
        events = self.event_factory.build()
        for e in events.values():
            self.rng.shuffle(e)

        schedule = Schedule(self.team_factory.build())
        self.teams = schedule.all_teams()
        self.rng.shuffle(self.teams)

        for r in self.config.rounds:
            rt = r.round_type
            evts = events.get(rt, [])
            if r.teams_per_round == r.rounds_per_team == 1:
                self._build_singles(schedule, rt, evts)
            else:
                self._build_matches(schedule, rt, evts)

        return schedule

    def _build_singles(self, schedule: Schedule, rt: RoundType, events: list[Event]) -> None:
        """Book all judging events for a specific round type."""
        teams = (t for t in self.teams if t.needs_round(rt))

        for event, team in zip(events, teams, strict=False):
            schedule.assign_single(event, team)

    def _build_matches(self, schedule: Schedule, rt: RoundType, events: list[Event]) -> None:
        """Book all events for a specific round type."""
        match_events = ((e, e.paired_event) for e in events if e.location.side == 1)

        for side1, side2 in match_events:
            available = (t for t in self.teams if t.needs_round(rt) and not t.conflicts(side1))
            if not (team1 := next(available, None)) or not (team2 := next(available, None)):
                continue
            schedule.assign_match(side1, side2, team1, team2)
