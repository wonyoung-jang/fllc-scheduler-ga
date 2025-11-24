"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..config.schemas import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.schedule import Schedule
    from ..fitness.fitness import HardConstraintChecker

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    rng: np.random.Generator
    checker: HardConstraintChecker
    repair_map: dict[int, Any] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
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

        teams, events = self.get_rt_tpr_maps(schedule)
        return self.iterative_repair(schedule, teams, events)

    def iterative_repair(
        self, schedule: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        repair_map = self.repair_map
        while len(schedule) < self.config.total_slots_required:
            filled = True
            for key, teams_for_rt in teams.items():
                _, tpr = key
                # if len(set(teams_for_rt)) < tpr:
                #     continue  # Not enough unique teams to fill a match, skip

                if not (events_for_rt := events.get(key)):
                    msg = f"No available events for round type {key[0]} with teams per round {tpr}"
                    raise ValueError(msg)

                if not (repair_fn := repair_map.get(tpr)):
                    msg = f"No assignment function for teams per round: {tpr}"
                    raise ValueError(msg)

                teams[key], events[key] = repair_fn(
                    teams=dict(enumerate(teams_for_rt)),
                    events=dict(enumerate(events_for_rt)),
                    schedule=schedule,
                )

                if teams[key]:  # noqa: PLR1733
                    filled = False
                    break

            if filled:
                return True

            event_indices = schedule.scheduled_events()
            event = self.rng.choice(event_indices)
            e_rt_idx = self.event_properties.roundtype_idx[event]
            ek = (e_rt_idx, self.config.round_idx_to_tpr[e_rt_idx])
            e1, e2 = event, None
            event_paired = self.event_properties.paired_idx[event]
            if event_paired != -1:
                if self.event_properties.loc_side[event] == 1:
                    e1, e2 = event, event_paired
                elif self.event_properties.loc_side[event] == 2:
                    e1, e2 = event_paired, event

            events[ek].append(e1)
            t1 = schedule[e1]
            teams[ek].append(t1)
            schedule.unassign(t1, e1)
            if e2 is not None:
                t2 = schedule[e2]
                teams[ek].append(t2)
                schedule.unassign(t2, e2)

        return len(schedule) == self.config.total_slots_required

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
        """Get the round type to team/player maps for the current schedule."""
        rt_tpr_config = self.config.round_idx_to_tpr

        teams: dict[tuple[int, int], list[int]] = defaultdict(list)
        for t, roundreqs in enumerate(schedule.team_rounds):
            for rt, n in enumerate(roundreqs):
                k = (rt, rt_tpr_config[rt])
                teams[k].extend([t] * n)

        _paired_idx = self.event_properties.paired_idx
        _loc_side = self.event_properties.loc_side
        _rt_idx = self.event_properties.roundtype_idx

        events: dict[tuple[int, int], list[int]] = defaultdict(list)
        for e in schedule.unscheduled_events():
            paired_e = _paired_idx[e]
            if (paired_e != -1 and _loc_side[e] == 1) or paired_e == -1:
                rt = _rt_idx[e]
                k = (rt, rt_tpr_config[rt])
                if k in teams:
                    events[k].append(e)

        return teams, events

    def repair_singles(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign single-team events to teams that need them."""
        while len(teams) >= 1:
            tkey = self.rng.choice(list(teams.keys()))
            t = teams.pop(tkey)

            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)

            for ekey in event_keys:
                e = events[ekey]
                if schedule.conflicts(t, e):
                    continue

                schedule.assign(t, e)
                events.pop(ekey)
                break
            else:
                teams[tkey] = t
                break

        return list(teams.values()), list(events.values())

    def repair_matches(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign match events to teams that need them."""
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

        # Handle case where odd number of teams and odd number of events required
        if len(teams) == 1 and events:
            tkey = next(iter(teams.keys()))
            t_solo = teams.pop(tkey)
            event_keys = list(events.keys())
            self.rng.shuffle(event_keys)
            for ekey in event_keys:
                e1 = events[ekey]
                if schedule.conflicts(t_solo, e1):
                    continue
                schedule.assign(t_solo, e1)
                events.pop(ekey)
                break
            else:
                teams[tkey] = t_solo

        return list(teams.values()), list(events.values())

    def find_and_repair_match(self, t1: int, t2: int, events: dict[int, int], schedule: Schedule) -> bool:
        """Find an open match slot for two teams and populate it."""
        _paired_idx = self.event_properties.paired_idx

        event_keys = list(events.keys())
        self.rng.shuffle(event_keys)

        for ekey in event_keys:
            e1 = events[ekey]
            e2 = _paired_idx[e1]
            if schedule.conflicts(t1, e1) or schedule.conflicts(t2, e2):
                continue

            schedule.assign(t1, e1)
            schedule.assign(t2, e2)
            events.pop(ekey)
            return True
        return False
