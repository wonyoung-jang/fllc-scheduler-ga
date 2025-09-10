"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    def repair(self, schedule: Schedule, to_destroy_count: int = 1) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        """
        if len(schedule) == self.config.total_slots:
            return True

        assign_map = {
            1: self.assign_singles,
            2: self.assign_matches,
        }

        teams_by_rt_tpr, events_by_rt_tpr = self.get_rt_tpr_maps(schedule)
        for key, teams_for_rt in teams_by_rt_tpr.items():
            _, tpr = key
            if len(set(teams_for_rt)) < tpr:
                break

            if not (events_for_rt := events_by_rt_tpr.get(key)):
                continue

            if not (assign_fn := assign_map.get(tpr)):
                continue

            assign_fn(
                teams=teams_for_rt,
                events=events_for_rt,
                schedule=schedule,
            )

        if len(schedule) == self.config.total_slots:
            schedule.clear_cache()
            return True

        events = list(schedule.keys())
        max_events_to_destroy = len(events) // 20  # 5% of events max
        for e in self.rng.sample(events, k=min(to_destroy_count, max_events_to_destroy)):
            schedule.destroy_event(e)

        schedule.clear_cache()
        return self.repair(schedule, to_destroy_count + 1)

    def get_rt_tpr_maps(self, schedule: Schedule) -> tuple[dict[int, list[Team]], dict[int, list[Event]]]:
        """Get the round type to team/player maps for the current schedule."""
        teams_by_rt_tpr: dict[tuple[int, int], dict[int, Team]] = defaultdict(list)
        for t in schedule.all_teams_needing_events():
            for rt, num_needed in ((rt, num) for rt, num in t.roundreqs.items() if num):
                key = (rt, self.rt_teams_needed[rt])
                teams_by_rt_tpr[key].extend(t for _ in range(num_needed))

        events_by_rt_tpr: dict[tuple[int, int], dict[int, Event]] = defaultdict(list)
        for e in self.set_of_events.difference(schedule.keys()):
            if (e.paired and e.location.side == 1) or e.paired is None:
                rt = e.roundtype
                key = (rt, self.rt_teams_needed[rt])
                if key in teams_by_rt_tpr:
                    events_by_rt_tpr[key].append(e)

        for rt_tpr in (teams_by_rt_tpr, events_by_rt_tpr):
            for key, values in rt_tpr.items():
                self.rng.shuffle(values)
                rt_tpr[key] = dict(enumerate(values))

        return teams_by_rt_tpr, events_by_rt_tpr

    def assign_singles(self, teams: dict[int, Team], events: dict[int, Event], schedule: Schedule) -> None:
        """Assign single-team events to teams that need them."""
        for t in teams.values():
            non_conflicting_events = ((i, e) for i, e in events.items() if not t.conflicts(e))
            for i, e in non_conflicting_events:
                schedule.assign_single(e, t)
                events.pop(i)
                break

    def assign_matches(self, teams: dict[int, Team], events: dict[int, Event], schedule: Schedule) -> None:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            t1 = teams.pop(self.rng.choice(list(teams.keys())))
            non_conflicting_teams = ((i, t2) for i, t2 in teams.items() if t1.identity != t2.identity)
            for i, t2 in non_conflicting_teams:
                if self.find_and_assign_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break

    def find_and_assign_match(self, t1: Team, t2: Team, events: dict[int, Event], schedule: Schedule) -> bool:
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
