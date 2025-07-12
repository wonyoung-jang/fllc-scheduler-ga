"""Builder for creating a valid schedule individual."""

import logging
import random
from dataclasses import dataclass, field

from ..config.config import RoundType, TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import Team, TeamFactory
from .fitness import FitnessEvaluator
from .schedule import Schedule
from .schedule_repairer import ScheduleRepairer

logger = logging.getLogger(__name__)


def create_and_evaluate_schedule(
    args: tuple[TeamFactory, EventFactory, TournamentConfig, FitnessEvaluator, ScheduleRepairer, int],
) -> Schedule | None:
    """Create and evaluate a schedule in a separate process."""
    team_factory, event_factory, config, evaluator, repairer, seed = args
    schedule = ScheduleBuilder(team_factory, event_factory, config, repairer, random.Random(seed)).build()
    if fitness_scores := evaluator.evaluate(schedule):
        schedule.fitness = fitness_scores
        return schedule
    return None


@dataclass(slots=True)
class ScheduleBuilder:
    """Encapsulates the logic for building a valid random schedule."""

    team_factory: TeamFactory
    event_factory: EventFactory
    config: TournamentConfig
    repairer: ScheduleRepairer
    rng: random.Random
    events: dict[RoundType, list[Event]] = field(init=False, repr=False)
    teams: list[Team] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.events = self.event_factory.build()

        for events in self.events.values():
            self.rng.shuffle(events)

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        schedule = Schedule(self.team_factory.build())
        self.teams = schedule.all_teams()
        self.rng.shuffle(self.teams)

        for r in self.config.rounds:
            if r.teams_per_round == r.rounds_per_team == 1:
                self._build_singles(schedule, r.round_type)
            else:
                self._build_matches(schedule, r.round_type)

        return schedule if schedule and self.repairer.repair(schedule) else None

    def _build_singles(self, schedule: Schedule, rt: RoundType) -> None:
        """Book all judging events for a specific round type."""
        events_for_round = self.events.get(rt, [])
        teams = (t for t in self.teams if t.needs_round(rt))

        for event, team in zip(events_for_round, teams, strict=False):
            schedule[event] = team

    def _build_matches(self, schedule: Schedule, rt: RoundType) -> None:
        """Book all events for a specific round type."""
        events_for_round = self.events.get(rt, [])
        teams = self.teams
        match_events = ((e, e.paired_event) for e in events_for_round if e.location.side == 1)

        for side1, side2 in match_events:
            available = (t for t in teams if t.needs_round(rt) and not t.conflicts(side1))

            if (team1 := next(available, None)) is None or (team2 := next(available, None)) is None:
                continue

            schedule.add_match(side1, side2, team1, team2)
