"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..config.schemas import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    rng: np.random.Generator
    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    rt_teams_needed: dict[int, int] = field(init=False, repr=False)
    repair_map: dict[int, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.rt_teams_needed = {rc.roundtype_idx: rc.teams_per_round for rc in self.config.rounds}
        self.repair_map = {
            1: self.repair_singles,
            2: self.repair_matches,
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
        teams_by_rt_tpr: dict[tuple[str, int], list[int]],
        events_by_rt_tpr: dict[tuple[str, int], list[int]],
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        if len(schedule) == self.config.total_slots_required:
            return True

        assign_map = self.repair_map
        for key, teams_for_rt in teams_by_rt_tpr.items():
            _, tpr = key
            if len(set(teams_for_rt)) < tpr:
                break  # Not enough unique teams to fill a match, recurse

            if not (events_for_rt := events_by_rt_tpr.get(key)):
                msg = f"No available events for round type {key[0]} with teams per round {tpr}"
                raise ValueError(msg)

            if not (assign_fn := assign_map.get(tpr)):
                msg = f"No assignment function for teams per round: {tpr}"
                raise ValueError(msg)

            teams_by_rt_tpr[key], events_by_rt_tpr[key] = assign_fn(
                teams=dict(enumerate(teams_for_rt)),
                events=dict(enumerate(events_for_rt)),
                schedule=schedule,
            )

            if teams_by_rt_tpr[key]:  # noqa: PLR1733
                break

        if len(schedule) != self.config.total_slots_required:
            event_indices = schedule.scheduled_events()
            event = self.rng.choice(event_indices)
            e_rt_idx = self.event_properties.roundtype_idx[event]

            ek = (e_rt_idx, self.rt_teams_needed[e_rt_idx])

            e1, e2 = event, None
            # if event.paired is not None:
            event_paired = self.event_properties.paired_idx[event]
            if event_paired != -1:
                if self.event_properties.loc_side[event] == 1:
                    e1, e2 = event, event_paired
                elif self.event_properties.loc_side[event] == 2:
                    e1, e2 = event_paired, event

            events_by_rt_tpr[ek].append(e1)

            t1 = schedule[e1]
            teams_by_rt_tpr[ek].append(t1)
            schedule.unassign(t1, e1)

            if e2 is not None:
                t2 = schedule[e2]
                teams_by_rt_tpr[ek].append(t2)
                schedule.unassign(t2, e2)

            return self.recursive_repair(schedule, teams_by_rt_tpr, events_by_rt_tpr)
        return True

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[str, int], list[int]], dict[tuple[str, int], list[int]]]:
        """Get the round type to team/player maps for the current schedule."""
        rt_tpr_config = self.rt_teams_needed

        teams_by_rt_tpr: dict[tuple[str, int], list[int]] = defaultdict(list)
        for t, roundreqs in enumerate(schedule.team_rounds):
            for rt, n in enumerate(roundreqs):
                k = (rt, rt_tpr_config[rt])
                teams_by_rt_tpr[k].extend([t] * n)

        _paired_idx = self.event_properties.paired_idx
        _loc_side = self.event_properties.loc_side
        _rt_idx = self.event_properties.roundtype_idx

        events_by_rt_tpr: dict[tuple[str, int], list[int]] = defaultdict(list)
        for e in schedule.unscheduled_events():
            paired_e = _paired_idx[e]
            if (paired_e != -1 and _loc_side[e] == 1) or paired_e == -1:
                rt = _rt_idx[e]
                k = (rt, rt_tpr_config[rt])
                if k in teams_by_rt_tpr:
                    events_by_rt_tpr[k].append(e)

        return teams_by_rt_tpr, events_by_rt_tpr

    def repair_singles(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign single-team events to teams that need them."""
        while len(teams) >= 1:
            tkey = self.rng.choice(list(teams.keys()))
            t = teams.pop(tkey)
            for i in self.rng.permutation(list(events.keys())):
                e = events[i]
                if schedule.conflicts(t, e):
                    continue

                schedule.assign(t, e)
                events.pop(i)
                break
            else:
                teams[tkey] = t
                break
        return list(teams.values()), list(events.values())

    def repair_matches(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign match events to teams that need them."""
        if len(teams) % 2 != 0:
            logger.debug("Odd number of teams (%d) for match assignment, one team will be left out.", len(teams))

        while len(teams) >= 2:
            tkey = self.rng.choice(list(teams.keys()))
            t1 = teams.pop(tkey)
            for i, t2 in teams.items():
                if t1 == t2:
                    continue

                if self.find_and_repair_match(t1, t2, events, schedule):
                    teams.pop(i)
                    break
            else:
                teams[tkey] = t1
                break

        return list(teams.values()), list(events.values())

    def find_and_repair_match(self, t1: int, t2: int, events: dict[int, int], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        _paired_idx = self.event_properties.paired_idx
        for i in self.rng.permutation(list(events.keys())):
            e1 = events[i]
            e2 = _paired_idx[e1]
            if schedule.conflicts(t1, e1) or schedule.conflicts(t2, e2):
                continue

            schedule.assign(t1, e1)
            schedule.assign(t2, e2)
            events.pop(i)
            return True
        return False
