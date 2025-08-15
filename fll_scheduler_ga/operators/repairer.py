"""A repairer for incomplete schedules."""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..config.config import TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.schedule import Schedule
from ..data_model.team import Team

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    rng: random.Random
    config: TournamentConfig
    event_factory: EventFactory
    set_of_events: set[Event] = field(init=False, repr=False)
    rt_teams_needed: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.set_of_events = set(self.event_factory.as_list())
        self.rt_teams_needed = {rc.roundtype: rc.teams_per_round for rc in self.config.rounds}

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        Args:
            schedule (Schedule): The schedule to repair.

        """
        if len(schedule) == self.config.total_slots:
            return True

        teams_per_round_map = {
            1: self._assign_singles,
            2: self._assign_matches,
        }

        evnts_by_rt_tpr = self._get_open_events(schedule)
        teams_by_rt_tpr = self._get_team_needs(schedule)

        for key, teams in teams_by_rt_tpr.items():
            if not (events := evnts_by_rt_tpr.get(key, [])):
                continue

            _, tpr = key
            if (assign := teams_per_round_map.get(tpr)) is None:
                continue

            assign(teams, events, schedule)

        return len(schedule) == self.config.total_slots

    def _get_open_events(self, schedule: Schedule) -> dict[tuple[str, int], list[Event]]:
        """Find all event slots not currently booked in the schedule."""
        evnts_by_rt_tpr = defaultdict(list)
        unbooked = list(self.set_of_events.difference(schedule.keys()))
        for e in self.rng.sample(unbooked, k=len(unbooked)):
            if (e.paired and e.location.side == 1) or e.paired is None:
                rt = e.roundtype
                key = (rt, self.rt_teams_needed[rt])
                evnts_by_rt_tpr[key].append(e)
        return evnts_by_rt_tpr

    def _get_team_needs(self, schedule: Schedule) -> dict[tuple[str, int], list[Team]]:
        """Determine which teams need which types of rounds."""
        teams_by_rt_tpr = defaultdict(list)
        teams = schedule.all_teams()
        for team in self.rng.sample(teams, k=len(teams)):
            for rt, num_needed in ((rt, num) for rt, num in team.roundreqs.items() if num):
                key = (rt, self.rt_teams_needed[rt])
                teams_by_rt_tpr[key].extend(team for _ in range(num_needed))
        return teams_by_rt_tpr

    def _assign_singles(self, teams: list[Team], events: list[Event], schedule: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        for team in teams:
            for i, event in enumerate(events):
                if team.conflicts(event):
                    continue

                schedule.assign_single(event, team)
                events.pop(i)
                break

    def _assign_matches(self, teams: list[Team], events: list[Event], schedule: Schedule) -> None:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            t1 = teams.pop(0)
            for i, t2 in enumerate(teams):
                if t1.identity == t2.identity:
                    continue

                if self._find_and_populate_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break

    def _find_and_populate_match(self, t1: Team, t2: Team, events: list[Event], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        for i, e1 in enumerate(events):
            if t1.conflicts(e1) or t2.conflicts(e1):
                continue

            schedule.assign_match(e1, e1.paired, t1, t2)
            events.pop(i)
            return True
        return False
