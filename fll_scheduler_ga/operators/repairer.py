"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..data_model.config import TournamentConfig
    from ..data_model.event import Event, EventFactory
    from ..data_model.schedule import Schedule
    from ..data_model.team import Team

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    rng: np.random.Generator
    config: TournamentConfig
    event_factory: EventFactory
    event_map: dict[int, Event] = None
    rt_teams_needed: dict[str, int] = field(init=False, repr=False)
    assign_map: dict[int, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.rt_teams_needed = {rc.roundtype: rc.teams_per_round for rc in self.config.rounds}
        self.event_map = self.event_factory.as_mapping()
        self.assign_map = {
            1: self.assign_singles,
            2: self.assign_matches,
        }

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        """
        if len(schedule) == self.config.total_slots_required:
            return True

        teams_by_rt_tpr, events_by_rt_tpr = self.get_rt_tpr_maps(schedule)
        return self.recursive_repair(schedule, teams_by_rt_tpr, events_by_rt_tpr)

    def recursive_repair(
        self,
        schedule: Schedule,
        teams_by_rt_tpr: dict[tuple[str, int], list[Team]],
        events_by_rt_tpr: dict[tuple[str, int], list[Event]],
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        if len(schedule) == self.config.total_slots_required:
            return True

        assign_map = self.assign_map
        for key, teams_for_rt in teams_by_rt_tpr.items():
            _, tpr = key
            if len({t.idx for t in teams_for_rt}) < tpr:
                break  # Not enough unique teams to fill a match, recurse

            if not (events_for_rt := events_by_rt_tpr.get(key)):
                msg = f"No available events for round type and teams per round: {key}"
                raise ValueError(msg)

            if not (assign_fn := assign_map.get(tpr)):
                msg = f"No assignment function for teams per round: {tpr}"
                raise ValueError(msg)

            teams_by_rt_tpr[key], events_by_rt_tpr[key] = assign_fn(
                teams=dict(enumerate(teams_for_rt)),
                events=dict(enumerate(events_for_rt)),
                schedule=schedule,
            )

            if teams_by_rt_tpr[key]:
                break

        if len(schedule) != self.config.total_slots_required:
            event_indices = schedule.scheduled_event_indices()
            event = self.event_map[self.rng.choice(event_indices)]

            ekey = (event.roundtype, self.rt_teams_needed[event.roundtype])
            e1, e2 = None, None
            if event.paired:
                if event.location.side == 1:
                    e1, e2 = event, event.paired
                elif event.location.side == 2:
                    e1, e2 = event.paired, event
            else:
                e1, e2 = event, None

            events_by_rt_tpr[ekey].append(e1)
            teams_by_rt_tpr[ekey].append(schedule[e1])
            if e2 is not None:
                teams_by_rt_tpr[ekey].append(schedule[e2])

            schedule.destroy_event(e1)
            return self.recursive_repair(schedule, teams_by_rt_tpr, events_by_rt_tpr)
        return True

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[str, int], list[Team]], dict[tuple[str, int], list[Event]]]:
        """Get the round type to team/player maps for the current schedule."""
        rt_tpr_config = self.rt_teams_needed
        teams_by_rt_tpr: dict[tuple[str, int], list[Team]] = defaultdict(list)
        for t in schedule.teams:
            for rt, n in ((rt, n) for rt, n in t.roundreqs.items() if n):
                k = (rt, rt_tpr_config[rt])
                teams_by_rt_tpr[k].extend(t for _ in range(n))

        events_by_rt_tpr: dict[tuple[str, int], list[Event]] = defaultdict(list)
        events_needed = schedule.unscheduled_event_indices()
        events_needed = [self.event_map[i] for i in events_needed]
        for e in events_needed:
            if (e.paired and e.location.side == 1) or e.paired is None:
                rt = e.roundtype
                k = (rt, rt_tpr_config[rt])
                if k in teams_by_rt_tpr:
                    events_by_rt_tpr[k].append(e)

        return teams_by_rt_tpr, events_by_rt_tpr

    def assign_singles(
        self, teams: dict[int, Team], events: dict[int, Event], schedule: Schedule
    ) -> tuple[list[Team], list[Event]]:
        """Assign single-team events to teams that need them."""
        while len(teams) >= 1:
            team_keys = list(teams.keys())
            tkey = self.rng.choice(team_keys)
            t = teams.pop(tkey)
            for i, e in events.items():
                if t.conflicts(e):
                    continue

                schedule.assign_single(e, t)
                events.pop(i)
                break
            else:
                teams[tkey] = t
                break

        return list(teams.values()), list(events.values())

    def assign_matches(
        self, teams: dict[int, Team], events: dict[int, Event], schedule: Schedule
    ) -> tuple[list[Team], list[Event]]:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            team_keys = list(teams.keys())
            tkey = self.rng.choice(team_keys)
            t1 = teams.pop(tkey)
            for i, t2 in teams.items():
                if t1.idx == t2.idx:
                    continue

                if self.find_and_assign_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break
            else:
                teams[tkey] = t1
                break

        return list(teams.values()), list(events.values())

    def find_and_assign_match(self, t1: Team, t2: Team, events: dict[int, Event], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        for i, e1 in events.items():
            e2 = e1.paired
            if t1.conflicts(e1) or t2.conflicts(e2):
                continue

            schedule.assign_match(e1, e2, t1, t2)
            events.pop(i)
            return True
        return False
