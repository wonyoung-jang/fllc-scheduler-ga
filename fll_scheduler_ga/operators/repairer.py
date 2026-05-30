"""A repairer for incomplete schedules."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from fll_scheduler_ga.domain.event import EventProperties
    from fll_scheduler_ga.domain.model import EventFactory, TournamentConfig
    from fll_scheduler_ga.domain.schedule import Schedule


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    rng: np.random.Generator
    checker: Callable[[Schedule], bool]
    repair_map: dict[int, Any] = field(init=False)
    _rt_to_tpr: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.repair_map = {1: self.repair_singles, 2: self.repair_matches}
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
        while schedule.get_size() < self.config.total_slots_required:
            if self._attempt_repair_step(teams, events, schedule):
                return True
            self._unassign_and_requeue_event(teams, events, schedule)
        return schedule.get_size() == self.config.total_slots_required

    def _attempt_repair_step(
        self, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]], schedule: Schedule
    ) -> bool:
        """Attempt to apply a repair function for the current round type.

        Returns True if the schedule is considered resolved for this step.
        """
        for key, teams_for_rt in teams.items():
            _, tpr = key
            if not (events_for_rt := events.get(key)):
                return True
            if not (repair_fn := self.repair_map.get(tpr)):
                msg = f"No assignment function for teams per round: {tpr}"
                raise ValueError(msg)
            _teams, _events = repair_fn(
                teams=dict(enumerate(teams_for_rt)), events=dict(enumerate(events_for_rt)), schedule=schedule
            )
            teams[key] = _teams
            events[key] = _events
            if _teams:
                return False
        return True

    def _unassign_and_requeue_event(
        self, teams: dict[tuple[int, int], list[int]], events: dict[tuple[int, int], list[int]], schedule: Schedule
    ) -> None:
        """Select a random scheduled event, handle pairing logic, and move it back to the queue."""
        event_indices = schedule.scheduled_events()
        self.rng.shuffle(event_indices)
        primary_event = event_indices[0]
        e_rt_idx = self.event_properties.roundtype_idx[primary_event]
        ek = (e_rt_idx, self.config.round_idx_to_tpr[e_rt_idx])
        paired_event = self.event_properties.paired_idx[primary_event]
        loc_side = self.event_properties.loc_side[primary_event]
        if paired_event != -1:
            e1, e2 = (paired_event, primary_event) if loc_side == 2 else (primary_event, paired_event)
        else:
            e1, e2 = primary_event, None
        t1 = schedule.schedule[e1]
        events[ek].append(e1)
        teams[ek].append(t1)
        schedule.unassign(t1, e1)
        if e2 is not None and (t2 := schedule.schedule[e2]) != -1:
            teams[ek].append(t2)
            schedule.unassign(t2, e2)

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
