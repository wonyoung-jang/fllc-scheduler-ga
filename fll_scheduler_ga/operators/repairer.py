"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config.app_schemas import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.schedule import Schedule
    from ..fitness.hard_constraint_checker import HardConstraintChecker

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    rng: np.random.Generator
    checker: HardConstraintChecker
    repair_map: dict[int, Any] = field(init=False)
    _rt_to_tpr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.repair_map = {
            1: self.repair_singles,
            2: self.repair_matches,
        }
        max_rt = max(self.config.round_idx_to_tpr.keys())
        self._rt_to_tpr = np.zeros(max_rt + 1, dtype=int)
        for rt, tpr in self.config.round_idx_to_tpr.items():
            self._rt_to_tpr[rt] = tpr

    def repair(self, schedule: Schedule) -> bool:
        """Repair missing assignments in the schedule.

        Fills in missing events for teams by assigning them to available (unbooked) event slots.

        """
        if schedule.get_size() == self.config.total_slots_required:
            return True

        teams, events = self.get_rt_tpr_maps(schedule)
        return self.iterative_repair(schedule, teams, events)

    def iterative_repair(
        self, schedule: Schedule, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]]
    ) -> bool:
        """Recursively repair the schedule by attempting to assign events to teams."""
        repair_map = self.repair_map
        while schedule.get_size() < self.config.total_slots_required:
            filled = True
            for key, teams_for_rt in teams.items():
                _, tpr = key

                if not (events_for_rt := events.get(key)):
                    break

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
            self.rng.shuffle(event_indices)
            event = event_indices[0]
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
            t1 = schedule.schedule[e1]
            teams[ek].append(t1)
            schedule.unassign(t1, e1)
            if e2 is not None:
                t2 = schedule.schedule[e2]
                if t2 != -1:
                    teams[ek].append(t2)
                    schedule.unassign(t2, e2)

        return schedule.get_size() == self.config.total_slots_required

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
        """Get the round type to team/player maps for the current schedule."""
        # 1. Team Map
        teams: dict[tuple[int, int], list[int]] = defaultdict(list)

        # Find (team_id, roundtype_id) where rounds are needed (>0)
        # team_rounds is shape (n_teams, n_round_types)
        t_idxs, rt_idxs = (schedule.team_rounds > 0).nonzero()

        if t_idxs.size > 0:
            # Get the counts (how many rounds needed)
            counts = schedule.team_rounds[t_idxs, rt_idxs]

            # If a team needs 2 rounds, we need 2 entries
            t_repeated = t_idxs.repeat(repeats=counts)  # ty:ignore[no-matching-overload]
            rt_repeated = rt_idxs.repeat(repeats=counts)  # ty:ignore[no-matching-overload]

            # Map roundtype to teams_per_round
            tpr_repeated = self._rt_to_tpr[rt_repeated]

            # Grouping by (rt, tpr)
            for i in range(len(t_repeated)):
                k = (rt_repeated[i], tpr_repeated[i])
                teams[k].append(t_repeated[i])

        # 2. Event Map
        events: dict[tuple[int, int], list[int]] = defaultdict(list)

        unscheduled = schedule.unscheduled_events()
        if unscheduled.size > 0:
            # Filter logic: (paired != -1 and side == 1) OR (paired == -1)
            paired = self.event_properties.paired_idx[unscheduled]
            sides = self.event_properties.loc_side[unscheduled]

            # Mask for valid repair candidates (singles or side 1 of matches)
            mask = (paired == -1) | (sides == 1)
            valid_events = unscheduled[mask]

            if valid_events.size > 0:
                valid_rts = self.event_properties.roundtype_idx[valid_events]
                valid_tprs = self._rt_to_tpr[valid_rts]

                for i in range(len(valid_events)):
                    k = (valid_rts[i], valid_tprs[i])
                    if k in teams:
                        events[k].append(valid_events[i])

        return teams, events

    def repair_singles(
        self, teams: dict[int, int], events: dict[int, int], schedule: Schedule
    ) -> tuple[list[int], list[int]]:
        """Assign single-team events to teams that need them."""
        while len(teams) >= 1:
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tkey = team_keys[0]
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
            team_keys = list(teams.keys())
            self.rng.shuffle(team_keys)
            tkey = team_keys[0]
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
