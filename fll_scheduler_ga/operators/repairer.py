"""A repairer for incomplete schedules."""

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from random import Random

from ..config.config import TournamentConfig
from ..data_model.event import Event, EventFactory
from ..data_model.schedule import Schedule
from ..data_model.team import Team

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    rng: Random
    config: TournamentConfig
    event_factory: EventFactory
    set_of_events: set[Event] = field(init=False, repr=False)
    rt_teams_needed: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.set_of_events = set(self.event_factory.build())
        self.rt_teams_needed = {rc.roundtype: rc.teams_per_round for rc in self.config.rounds}

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        Args:
            schedule (Schedule): The schedule to repair.

        """
        if len(schedule) == self.config.total_slots:
            return True

        tpr_assign_map = {
            1: Repairer._assign_singles,
            2: self._assign_matches,
        }

        teams_by_rt_tpr: dict[tuple[int, int], dict[int, Team]] = defaultdict(list)
        teams = schedule.all_teams()
        for team in self.rng.sample(teams, k=len(teams)):
            for rt, num_needed in ((rt, num) for rt, num in team.roundreqs.items() if num):
                key = (rt, self.rt_teams_needed[rt])
                teams_by_rt_tpr[key].extend(team for _ in range(num_needed))

        events_by_rt_tpr: dict[tuple[int, int], dict[int, Event]] = defaultdict(list)
        events = self.set_of_events.difference(schedule.keys())
        for e in events:
            if (e.paired and e.location.side == 1) or e.paired is None:
                rt = e.roundtype
                key = (rt, self.rt_teams_needed[rt])
                if key in teams_by_rt_tpr:
                    events_by_rt_tpr[key].append(e)

        for dictionary in (teams_by_rt_tpr, events_by_rt_tpr):
            for key, value in dictionary.items():
                dictionary[key] = dict(enumerate(value))

        for key, teams_for_rt in teams_by_rt_tpr.items():
            _, tpr = key
            if len(set(teams_for_rt)) < tpr:
                break

            events_for_rt = events_by_rt_tpr.get(key)
            assign = tpr_assign_map.get(tpr)
            if events_for_rt and assign:
                assign(
                    teams=teams_for_rt,
                    events=events_for_rt,
                    schedule=schedule,
                )

        schedule.clear_cache()
        return len(schedule) == self.config.total_slots

    @staticmethod
    def _assign_singles(teams: dict[int, Team], events: dict[int, Event], schedule: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        for team in teams.values():
            non_conflicting_events = ((i, e) for i, e in events.items() if not team.conflicts(e))
            for i, event in non_conflicting_events:
                schedule.assign_single(event, team)
                events.pop(i)
                break

    def _assign_matches(self, teams: dict[int, Team], events: dict[int, Event], schedule: Schedule) -> None:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            t1 = teams.pop(self.rng.choice(list(teams.keys())))
            non_conflicting_teams = ((i, t2) for i, t2 in teams.items() if t1.identity != t2.identity)
            for i, t2 in non_conflicting_teams:
                if Repairer._find_and_populate_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break

    @staticmethod
    def _find_and_populate_match(t1: Team, t2: Team, events: dict[int, Event], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        non_conflicting_events = (
            (i, e, e.paired)
            for i, e in events.items()
            if e.paired and not t1.conflicts(e) and not t2.conflicts(e.paired)
        )
        for i, e1, e2 in non_conflicting_events:
            schedule.assign_match(e1, e2, t1, t2)
            events.pop(i)
            return True
        return False
