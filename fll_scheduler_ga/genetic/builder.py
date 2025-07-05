"""Builder for creating a valid schedule individual."""

import logging
import random
from dataclasses import dataclass, field

from ..config.config import Round, RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import TeamFactory
from .schedule import Schedule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Encapsulates the logic for building a valid random schedule."""

    _team_factory: TeamFactory
    _event_factory: EventFactory
    _config: TournamentConfig
    _rng: random.Random
    _schedule: Schedule = field(init=False, repr=False)
    _events: dict[RoundType, list[Event]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._events = self._event_factory.build()

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        self._schedule = Schedule(self._team_factory.build())

        for r in self._config.rounds:
            if r.teams_per_round == 1:
                self._book_judging_rounds(r)
            else:
                self._book_rounds(r)

        return self._schedule

    def _book_judging_rounds(self, r: Round) -> None:
        """Book all judging events for a specific round type."""
        events_for_round = list(self._events.get(r.round_type, []))
        teams_needing_round = [t for t in self._schedule.all_teams if t.needs_round(r.round_type)]

        self._rng.shuffle(events_for_round)
        self._rng.shuffle(teams_needing_round)

        for event in events_for_round:
            for i, t in enumerate(teams_needing_round):
                if not t.conflicts(event):
                    t.add_event(event)
                    self._schedule[event] = t
                    teams_needing_round.pop(i)
                    break

    def _book_rounds(self, r: Round) -> None:
        """Book all events for a specific round type."""
        events_for_round = list(self._events.get(r.round_type, []))
        self._rng.shuffle(events_for_round)
        all_teams = self._schedule.all_teams

        for side1 in events_for_round:
            if side1.location.side != 1:
                continue

            side2 = side1.paired_event

            available_for_side1 = [t for t in all_teams if t.needs_round(r.round_type) and not t.conflicts(side1)]
            if not available_for_side1:
                continue

            teams_with_location_side1 = [t for t in available_for_side1 if t.has_location(side1)]
            if len(teams_with_location_side1) >= 1:
                available_for_side1 = teams_with_location_side1

            team1 = self._rng.choice(available_for_side1)

            available_for_side2 = [
                t
                for t in all_teams
                if t.identity != team1.identity and t.needs_round(r.round_type) and not t.conflicts(side2)
            ]
            if not available_for_side2:
                continue

            teams_with_location_side2 = [t for t in available_for_side2 if t.has_location(side2)]
            if len(teams_with_location_side2) >= 1:
                available_for_side2 = teams_with_location_side2

            team2 = self._rng.choice(available_for_side2)
            team1.add_event(side1)
            team2.add_event(side2)
            team1.add_opponent(team2)
            team2.add_opponent(team1)
            self._schedule[side1] = team1
            self._schedule[side2] = team2
