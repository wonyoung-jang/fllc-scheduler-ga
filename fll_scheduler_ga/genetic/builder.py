"""Builder for creating a valid schedule individual."""

import logging
import random
from dataclasses import dataclass, field

from ..config.config import Round, RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import TeamFactory
from .schedule import Schedule
from .schedule_repairer import ScheduleRepairer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Encapsulates the logic for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    rng: random.Random
    schedule: Schedule = field(init=False, repr=False)
    events: dict[RoundType, list[Event]] = field(init=False, repr=False)
    repairer: ScheduleRepairer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.events = self.event_factory.build()
        self.repairer = ScheduleRepairer(self.event_factory, self.rng)

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        self.schedule = Schedule(self.team_factory.build())

        for r in self.config.rounds:
            if r.teams_per_round == 1:
                self._book_judging_rounds(r)
            else:
                self._book_rounds(r)

        return self.schedule if self.schedule and self.repairer.repair(self.schedule) else None

    def _book_judging_rounds(self, r: Round) -> None:
        """Book all judging events for a specific round type."""
        events_for_round = self.events.get(r.round_type, [])
        teams_needing_round = [t for t in self.schedule.all_teams() if t.needs_round(r.round_type)]

        self.rng.shuffle(events_for_round)
        self.rng.shuffle(teams_needing_round)

        for event in events_for_round:
            for i, t in enumerate(teams_needing_round):
                if not t.conflicts(event):
                    self.schedule[event] = t
                    teams_needing_round.pop(i)
                    break

    def _book_rounds(self, r: Round) -> None:
        """Book all events for a specific round type."""
        events_for_round = self.events.get(r.round_type, [])
        self.rng.shuffle(events_for_round)
        teams = self.schedule.all_teams()

        events = ((e, e.paired_event) for e in events_for_round if e.location.side == 1)

        for side1, side2 in events:
            available = [t for t in teams if t.needs_round(r.round_type) and not t.conflicts(side1)]

            if len(available) < 2:
                continue

            team1, team2 = self.rng.sample(available, 2)

            self.schedule.add_match(side1, side2, team1, team2)
