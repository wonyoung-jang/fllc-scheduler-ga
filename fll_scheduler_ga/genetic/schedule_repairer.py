# fll_scheduler_ga/genetic/repairer.py
"""A repairer for incomplete schedules."""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..config.config import TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import Team
from .schedule import Schedule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduleRepairer:
    """Class to handle the repair of schedules with missing event assignments."""

    event_factory: EventFactory
    rng: random.Random
    config: TournamentConfig = field(init=False)
    all_events: set[Event] = field(init=False)
    rt_teams_needed: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.config = self.event_factory.config
        self.all_events = set(self.event_factory.flat_list())
        self.rt_teams_needed = {rc.round_type: rc.teams_per_round for rc in self.config.rounds}

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available
        (un-booked) event slots.

        Args:
            schedule (Schedule): The schedule to repair.

        Returns:
            bool: True if the schedule was successfully repaired (all teams have all
                  required events), False otherwise.

        """
        open_events = self._get_open_events(schedule)
        needs_by_rt = self._get_team_needs(schedule)

        for (rt, tpr), teams in needs_by_rt.items():
            if tpr == 1:
                self._assign_singles(teams, open_events.get((rt, tpr), []), schedule)
            elif tpr == 2:
                self._assign_matches(teams, open_events.get((rt, tpr), []), schedule)

        return all(t.rounds_needed() == 0 for t in schedule.all_teams())

    def _get_open_events(self, schedule: Schedule) -> defaultdict[tuple[str, int], list[Event]]:
        """Find all event slots not currently booked in the schedule."""
        open_events = defaultdict(list)
        unbooked_events = self.all_events.difference(schedule.keys())

        for e in unbooked_events:
            if (e.paired_event is not None and e.location.side == 1) or e.paired_event is None:
                key = (e.round_type, self.rt_teams_needed[e.round_type])
                open_events[key].append(e)

        for events in open_events.values():
            self.rng.shuffle(events)

        return open_events

    def _get_team_needs(self, schedule: Schedule) -> defaultdict[tuple[str, int], list[Team]]:
        """Determine which teams need which types of rounds."""
        needs_by_rt = defaultdict(list)

        for team in schedule.all_teams():
            for rt, num_needed in team.round_types.items():
                if num_needed > 0:
                    key = (rt, self.rt_teams_needed[rt])
                    for _ in range(num_needed):
                        needs_by_rt[key].append(team)

        return needs_by_rt

    def _assign_singles(self, teams: list[Team], open_events: list[Event], schedule: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        self.rng.shuffle(teams)

        for team in teams:
            for i, event in enumerate(open_events):
                if not team.conflicts(event):
                    schedule[event] = team
                    open_events.pop(i)
                    break

    def _assign_matches(self, teams: list[Team], open_events: list[Event], schedule: Schedule) -> None:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        self.rng.shuffle(teams)
        team_pool = list(teams)

        while len(team_pool) >= 2:
            team1 = team_pool.pop(0)
            partner_found = False

            for j, team2 in enumerate(team_pool):
                if team1.identity != team2.identity:
                    self._find_and_populate_match(team1, team2, open_events, schedule)
                    team_pool.pop(j)
                    partner_found = True
                    break

            if not partner_found:
                logger.debug("Could not find a match partner for team %d", team1.identity)

    def _find_and_populate_match(self, t1: Team, t2: Team, open_events: list[Event], schedule: Schedule) -> None:
        """Find an open match slot for two teams and populate it."""
        for i, e1 in enumerate(open_events):
            e2 = e1.paired_event

            if not t1.conflicts(e1) and not t2.conflicts(e2):
                schedule.add_match(e1, e2, t1, t2)
                open_events.pop(i)
                return

            if not t1.conflicts(e2) and not t2.conflicts(e1):
                schedule.add_match(e2, e1, t1, t2)
                open_events.pop(i)
                return
