# fll_scheduler_ga/genetic/repairer.py
"""A repairer for incomplete schedules."""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..config.config import TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.team import Team
from ..genetic.schedule import Schedule

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
        self.set_of_events = set(self.event_factory.flat_list())
        self.rt_teams_needed = {rc.round_type: rc.teams_per_round for rc in self.config.rounds}

    def repair(self, schedule: Schedule) -> Schedule | None:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        Args:
            schedule (Schedule): The schedule to repair.

        """
        open_events = self._get_open_events(schedule)
        needs_by_rt = self._get_team_needs(schedule)

        for (rt, tpr), teams in needs_by_rt.items():
            slots = open_events.get((rt, tpr), [])
            if tpr == 1:
                self._assign_singles(teams, slots, schedule)
            elif tpr == 2:
                self._assign_matches(teams, slots, schedule)

        return schedule if len(schedule) == self.config.total_slots else None

    def _get_open_events(self, schedule: Schedule) -> dict[tuple[str, int], list[Event]]:
        """Find all event slots not currently booked in the schedule."""
        open_events = defaultdict(list)
        unbooked = self.set_of_events.difference(schedule.keys())

        for e in unbooked:
            if (e.paired_event and e.location.side == 1) or e.paired_event is None:
                key = (e.round_type, self.rt_teams_needed[e.round_type])
                open_events[key].append(e)

        for events in open_events.values():
            self.rng.shuffle(events)

        return open_events

    def _get_team_needs(self, schedule: Schedule) -> dict[tuple[str, int], list[Team]]:
        """Determine which teams need which types of rounds."""
        needs_by_rt = defaultdict(list)

        for team in schedule.all_teams():
            for rt, num_needed in team.round_types.items():
                if num_needed > 0:
                    key = (rt, self.rt_teams_needed[rt])
                    needs_by_rt[key].extend([team] * num_needed)

        for teams in needs_by_rt.values():
            self.rng.shuffle(teams)

        return needs_by_rt

    def _assign_singles(self, teams: list[Team], events: list[Event], schedule: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        for team in teams:
            for i, event in enumerate(events):
                if team.conflicts(event):
                    continue

                team.add_event(event)
                schedule[event] = team
                events.pop(i)
                break

    def _assign_matches(self, teams: list[Team], events: list[Event], schedule: Schedule) -> None:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            team1 = teams.pop(0)
            partner_found = False

            for j, team2 in enumerate(teams):
                if team1.identity == team2.identity:
                    continue

                if not self._find_and_populate_match(team1, team2, events, schedule):
                    break

                teams.pop(j)
                partner_found = True
                break

            if not partner_found:
                logger.debug("Could not find a match partner for team %d", team1.identity)

    def _find_and_populate_match(self, t1: Team, t2: Team, events: list[Event], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        for i, e1 in enumerate(events):
            if t1.conflicts(e1):
                continue

            e2 = e1.paired_event

            if t2.conflicts(e2):
                continue

            t1.add_event(e1)
            t2.add_event(e2)
            schedule[e1] = t1
            schedule[e2] = t2
            t1.add_opponent(t2)
            t2.add_opponent(t1)
            events.pop(i)
            return True

        return False
