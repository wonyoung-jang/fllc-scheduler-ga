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

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        Args:
            schedule (Schedule): The schedule to repair.

        """
        if len(schedule) == self.config.total_slots:
            return True

        open_events = self._get_open_events(schedule)
        needs_by_rt = self._get_team_needs(schedule)

        for (rt, tpr), teams in needs_by_rt.items():
            slots = open_events.get((rt, tpr), [])
            if tpr == 1:
                self._assign_singles(teams, slots, schedule)
            elif tpr == 2:
                self._assign_matches(teams, slots, schedule)

        return len(schedule) == self.config.total_slots

    def _get_open_events(self, schedule: Schedule) -> dict[tuple[str, int], list[Event]]:
        """Find all event slots not currently booked in the schedule."""
        open_events = defaultdict(list)
        unbooked = self.set_of_events.difference(schedule.keys())

        for e in self.rng.sample(list(unbooked), k=len(unbooked)):
            if (e.paired_event and e.location.side == 1) or e.paired_event is None:
                rt = e.round_type
                key = (rt, self.rt_teams_needed[rt])
                open_events[key].append(e)

        return open_events

    def _get_team_needs(self, schedule: Schedule) -> dict[tuple[str, int], list[Team]]:
        """Determine which teams need which types of rounds."""
        needs_by_rt = defaultdict(list)
        teams = schedule.all_teams()

        for team in self.rng.sample(teams, k=len(teams)):
            for rt, num_needed in team.round_types.items():
                if num_needed > 0:
                    key = (rt, self.rt_teams_needed[rt])
                    needs_by_rt[key].extend([team] * num_needed)

        return needs_by_rt

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
            partner_idx = -1
            for i, t2 in enumerate(teams):
                if t1.identity == t2.identity:
                    continue

                if self._find_and_populate_match(t1, t2, events, schedule):
                    partner_idx = i
                    break

            if partner_idx != -1:
                teams.pop(partner_idx)
            else:
                logger.debug("Could not find a match partner for team %d", t1.identity)

    def _find_and_populate_match(self, t1: Team, t2: Team, events: list[Event], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        event_idx = -1
        for i, e1 in enumerate(events):
            if t1.conflicts(e1):
                continue

            e2 = e1.paired_event

            if t2.conflicts(e2):
                continue

            schedule.assign_match(e1, e2, t1, t2)
            event_idx = i
            break

        if event_idx != -1:
            events.pop(event_idx)
            return True
        return False
