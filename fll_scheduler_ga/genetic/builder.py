"""Builder for creating a valid schedule individual."""

from dataclasses import dataclass
from logging import getLogger
from random import Random

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.schedule import Schedule
from ..data_model.team import Team, TeamFactory

logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    rng: Random

    def build(self, rng: Random | None = None) -> Schedule:
        """Construct and return the final schedule."""
        rng = self.rng if rng is None else rng
        events = self.event_factory.build()
        for rt in events:
            rng.shuffle(events[rt])

        schedule = Schedule(self.team_factory.build())
        teams = schedule.all_teams()
        teams = rng.sample(teams, k=len(teams))

        _build_map = {
            1: self._build_singles,
            2: self._build_matches,
        }

        for r in self.config.rounds:
            rt = r.roundtype
            evts = events.get(rt, [])
            build_fn = _build_map.get(r.teams_per_round, None)
            if build_fn:
                build_fn(schedule, rt, evts, teams)

        schedule.clear_cache()
        return schedule

    def _build_singles(self, schedule: Schedule, rt: RoundType, events: list[Event], teams: list[Team]) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            available = (t for t in teams if t.needs_round(rt) and not t.conflicts(event))
            team = next(available, None)
            if team:
                schedule.assign_single(event, team)

    def _build_matches(self, schedule: Schedule, rt: RoundType, events: list[Event], teams: list[Team]) -> None:
        """Book all events for a specific round type."""
        for side1, side2 in ((e, e.paired) for e in events if e.location.side == 1):
            available = (t for t in teams if t.needs_round(rt) and not t.conflicts(side1))
            team1 = next(available, None)
            team2 = next(available, None)
            if team1 and team2:
                schedule.assign_match(side1, side2, team1, team2)
