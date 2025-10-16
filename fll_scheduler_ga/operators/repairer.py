"""A repairer for incomplete schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ..data_model.config import TournamentConfig
    from ..data_model.event import Event, EventFactory, EventProperties
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class Repairer:
    """Class to handle the repair of schedules with missing event assignments."""

    rng: np.random.Generator
    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties
    event_map: dict[int, Event] = None
    rt_teams_needed: dict[int, int] = field(init=False, repr=False)
    repair_map: dict[int, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.rt_teams_needed = {rc.roundtype_idx: rc.teams_per_round for rc in self.config.rounds}
        self.event_map = self.event_factory.as_mapping()
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
        events_by_rt_tpr: dict[tuple[str, int], list[Event]],
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
            event_idx = self.rng.choice(event_indices)
            event = self.event_map[event_idx]
            event_rti = event.roundtype_idx

            ek = (event_rti, self.rt_teams_needed[event_rti])

            e1, e2 = event, None
            if event.paired is not None:
                if event.location.side == 1:
                    e1, e2 = event, event.paired
                elif event.location.side == 2:
                    e1, e2 = event.paired, event

            events_by_rt_tpr[ek].append(e1)

            t1 = schedule[e1.idx]
            teams_by_rt_tpr[ek].append(t1)
            schedule.unassign(t1, e1.idx)

            if e2 is not None:
                t2 = schedule[e2.idx]
                teams_by_rt_tpr[ek].append(t2)
                schedule.unassign(t2, e2.idx)

            return self.recursive_repair(schedule, teams_by_rt_tpr, events_by_rt_tpr)
        return True

    def get_rt_tpr_maps(
        self, schedule: Schedule
    ) -> tuple[dict[tuple[str, int], list[int]], dict[tuple[str, int], list[Event]]]:
        """Get the round type to team/player maps for the current schedule."""
        rt_tpr_config = self.rt_teams_needed

        teams_by_rt_tpr: dict[tuple[str, int], list[int]] = defaultdict(list)
        for t, roundreqs in enumerate(schedule.team_rounds):
            for rt, n in enumerate(roundreqs):
                k = (rt, rt_tpr_config[rt])
                teams_by_rt_tpr[k].extend([t] * n)

        events_by_rt_tpr: dict[tuple[str, int], list[Event]] = defaultdict(list)
        for e in self.event_map.values():
            if schedule[e.idx] != -1:
                continue

            if (e.paired is not None and e.location.side == 1) or e.paired is None:
                rt = e.roundtype_idx
                k = (rt, rt_tpr_config[rt])
                if k in teams_by_rt_tpr:
                    events_by_rt_tpr[k].append(e)

        return teams_by_rt_tpr, events_by_rt_tpr

    def repair_singles(
        self,
        teams: dict[int, int],
        events: dict[int, Event],
        schedule: Schedule,
    ) -> tuple[list[int], list[Event]]:
        """Assign single-team events to teams that need them."""
        while len(teams) >= 1:
            tkey = self.rng.choice(list(teams.keys()))
            t = teams.pop(tkey)
            for i in self.rng.permutation(list(events.keys())):
                e = events[i]
                if schedule.conflicts(t, e.idx):
                    continue

                schedule.assign(t, e.idx)
                events.pop(i)
                break
            else:
                teams[tkey] = t
                break
        return list(teams.values()), list(events.values())

    def repair_matches(
        self,
        teams: dict[int, int],
        events: dict[int, Event],
        schedule: Schedule,
    ) -> tuple[list[int], list[Event]]:
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

    def find_and_repair_match(
        self,
        t1: int,
        t2: int,
        events: dict[int, Event],
        schedule: Schedule,
    ) -> bool:
        """Find an open match slot for two teams and populate it."""
        for i in self.rng.permutation(list(events.keys())):
            e1 = events[i]
            e2 = e1.paired
            if schedule.conflicts(t1, e1.idx) or schedule.conflicts(t2, e2.idx):
                continue

            schedule.assign(t1, e1.idx)
            schedule.assign(t2, e2.idx)
            events.pop(i)
            return True
        return False


# @dataclass(slots=True)
# class Repairer:
#     """Class to handle the repair of schedules with missing event assignments."""

#     rng: np.random.Generator
#     config: TournamentConfig
#     event_factory: EventFactory
#     event_properties: EventProperties
#     event_map: dict[int, Event] = None
#     rt_teams_needed: dict[int, int] = field(init=False, repr=False)
#     repair_map: dict[int, Any] = field(init=False, repr=False)
#     roundtype_events: dict[int, list[int]] = None

#     def __post_init__(self) -> None:
#         """Post-initialization to set up the initial state."""
#         self.rt_teams_needed = {rc.roundtype_idx: rc.teams_per_round for rc in self.config.rounds}
#         self.event_map = self.event_factory.as_mapping()
#         self.repair_map = {
#             1: self.repair_singles,
#             2: self.repair_matches,
#         }
#         self.roundtype_events = self.event_factory.as_roundtype_indices()

#     def repair(self, schedule: Schedule) -> bool:
#         """Repair missing assignments in the schedule.

#         Fills in missing events for teams by assigning them to available (unbooked) event slots.

#         """
#         if len(schedule) == self.config.total_slots_required:
#             return True

#         return self.recursive_repair(schedule)

#     def recursive_repair(self, schedule: Schedule) -> bool:
#         """Recursively repair the schedule by attempting to assign events to teams."""
#         events = dict(self.roundtype_events.items())
#         unscheduled_events = schedule.unscheduled_events()
#         for rt, evts in events.items():
#             unscheduled = np.intersect1d(unscheduled_events, evts, assume_unique=True)
#             if self.config.round_idx_to_tpr[rt] == 1:
#                 self.repair_singles(schedule, unscheduled, rt)
#             elif self.config.round_idx_to_tpr[rt] == 2:
#                 self.repair_matches(schedule, unscheduled, rt)

#         if len(schedule) == self.config.total_slots_required:
#             return True

#         event_indices = schedule.scheduled_events()
#         event = self.rng.choice(event_indices)

#         e1, e2 = event, None
#         if self.event_properties.paired_idx[event] != -1:
#             if self.event_properties.loc_side[event] == 1:
#                 e1, e2 = event, self.event_properties.paired_idx[event]
#             elif self.event_properties.loc_side[event] == 2:
#                 e1, e2 = self.event_properties.paired_idx[event], event

#         t1 = schedule[e1]
#         schedule.unassign(t1, e1)

#         if e2 is not None:
#             t2 = schedule[e2]
#             schedule.unassign(t2, e2)

#         return self.recursive_repair(schedule)

#     def repair_singles(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
#         """Book all judging events for a specific round type."""
#         for event in self.rng.permutation(events):
#             needs_rounds = schedule.all_rounds_needed(roundtype)
#             self.rng.shuffle(needs_rounds)
#             available = (t for t in needs_rounds if not schedule.conflicts(t, event))
#             team = next(available, None)
#             if team:
#                 schedule.assign(team, event)

#     def repair_matches(self, schedule: Schedule, events: np.ndarray, roundtype: int) -> None:
#         """Book all events for a specific round type."""
#         for e1 in self.rng.permutation(events):
#             e2 = self.event_properties.paired_idx[e1]
#             if e2 == -1 or self.event_properties.loc_side[e1] != 1:
#                 continue
#             needs_rounds = schedule.all_rounds_needed(roundtype)
#             self.rng.shuffle(needs_rounds)
#             available = (t for t in needs_rounds if not schedule.conflicts(t, e1) and not schedule.conflicts(t, e2))
#             t1 = next(available, None)
#             t2 = next(available, None)
#             if t1 and t2:
#                 schedule.assign(t1, e1)
#                 schedule.assign(t2, e2)
