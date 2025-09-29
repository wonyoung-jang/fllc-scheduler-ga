"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import MutationOp

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from ..config.app_config import AppConfig
    from ..data_model.config import RoundType
    from ..data_model.event import Event, EventFactory
    from ..data_model.schedule import Match, Schedule
    from ..data_model.team import Team
    from ..data_model.time import TimeSlot

logger = getLogger(__name__)


def build_mutations(app_config: AppConfig, event_factory: EventFactory) -> Iterator[Mutation]:
    """Build and return a tuple of mutation operators based on the configuration."""
    rng = app_config.rng
    mutations = {
        # SwapMatchMutation variants
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda: SwapMatchMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda: SwapMatchMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda: SwapMatchMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTeamMutation variants
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda: SwapTeamMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda: SwapTeamMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda: SwapTeamMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTableSideMutation variant
        MutationOp.SWAP_TABLE_SIDE: lambda: SwapTableSideMutation(
            rng=rng,
            event_factory=event_factory,
            same_timeslot=True,
            same_location=True,
        ),
        # TimeSlotSequenceMutation variants
        MutationOp.INVERSION: lambda: InversionMutation(
            rng=rng,
            event_factory=event_factory,
        ),
        MutationOp.SCRAMBLE: lambda: ScrambleMutation(
            rng=rng,
            event_factory=event_factory,
        ),
    }

    if not (mutation_types := app_config.operators.mutation_types):
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return

    for mutation_name in mutation_types:
        if mutation_name not in mutations:
            msg = f"Unknown mutation type in config: '{mutation_name}'"
            raise ValueError(msg)
        else:
            yield mutations[mutation_name]()


@dataclass(slots=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: np.random.Generator
    event_factory: EventFactory

    @abstractmethod
    def mutate(self, schedule: Schedule) -> bool:
        """Mutate a schedule schedule to introduce genetic diversity.

        Args:
            schedule (Schedule): The schedule to mutate.

        Returns:
            bool: True if mutation was successful, False otherwise.

        """
        msg = "Mutate method must be implemented by subclasses."
        raise NotImplementedError(msg)


@dataclass(slots=True)
class SwapMutation(Mutation):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    same_timeslot: bool = False
    same_location: bool = False
    swap_candidates: list[tuple[Event, ...]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.swap_candidates = self.init_swap_candidates()

    @abstractmethod
    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match | None]:
        """Get candidates for swapping teams in the schedule schedule.

        Args:
            schedule (Schedule): The schedule to analyze.

        Returns:
            tuple[Match | None]: A tuple containing two matches to swap,
            or None if no valid candidates are found.

        """
        msg = "get_swap_candidates method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def init_swap_candidates(self) -> list[tuple[Event, ...]]:
        """Precompute any necessary data before mutation."""
        swap_candidates = []
        for match_list in self.event_factory.as_matches().values():
            for match1, match2 in itertools.combinations(match_list, 2):
                e1, _ = match1
                e2, _ = match2
                is_same_timeslot = e1.timeslot.idx == e2.timeslot.idx
                is_same_location = e1.location.idx == e2.location.idx

                if not (self.same_timeslot and self.same_location) and not self._validate_swap(
                    is_same_timeslot=is_same_timeslot,
                    is_same_location=is_same_location,
                    same_timeslot=self.same_timeslot,
                    same_location=self.same_location,
                ):
                    continue

                swap_candidates.append((match1, match2))
        logger.debug("Initialized %d swap candidates", len(swap_candidates))
        return swap_candidates

    def _validate_swap(
        self,
        *,
        is_same_timeslot: bool,
        is_same_location: bool,
        same_timeslot: bool,
        same_location: bool,
    ) -> bool:
        """Check if the swap between two events is valid based on timeslot and location."""
        timeslot_condition = is_same_timeslot == same_timeslot
        location_condition = is_same_location == same_location
        return timeslot_condition and location_condition


@dataclass(slots=True)
class SwapTeamMutation(SwapMutation):
    """Mutation operator for swapping single team between two matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_TEAM_SAME_LOCATION,
            (False, True): MutationOp.SWAP_TEAM_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

    def mutate(self, schedule: Schedule) -> bool:
        """Swap one team from two different matches."""
        match1_data, match2_data = self.get_swap_candidates(schedule)

        if match1_data is None:
            return False

        e1a, _, t1a, _ = match1_data
        e2a, _, t2a, _ = match2_data

        t1a.switch_event(e1a, e2a)
        t2a.switch_event(e2a, e1a)

        schedule[e1a] = t2a
        schedule[e2a] = t1a

        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...]:
        """Get two matches to swap in the schedule schedule."""
        if not self.swap_candidates:
            return None

        n_candidates = len(self.swap_candidates)
        for idx in self.rng.permutation(n_candidates):
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data

            t1a, t1b = schedule[e1a], schedule[e1b]
            t2a, t2b = schedule[e2a], schedule[e2b]
            if None in (t1a, t1b, t2a, t2b):
                continue

            match_team_ids = {t1a.idx, t1b.idx, t2a.idx, t2b.idx}
            if len(match_team_ids) < 4 or t1a.conflicts(e2a, ignore=e1a) or t2a.conflicts(e1a, ignore=e2a):
                continue

            return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)

        return None, None


@dataclass(slots=True)
class SwapMatchMutation(SwapMutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_MATCH_SAME_LOCATION,
            (False, True): MutationOp.SWAP_MATCH_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

    def mutate(self, schedule: Schedule) -> bool:
        """Swap two entire matches."""
        match1_data, match2_data = self.get_swap_candidates(schedule)

        if match1_data is None:
            return False

        e1a, e1b, t1a, t1b = match1_data
        e2a, e2b, t2a, t2b = match2_data
        is_match1_teams = None not in (t1a, t1b)
        is_match2_teams = None not in (t2a, t2b)

        if not is_match1_teams and not is_match2_teams:
            return False

        if is_match1_teams:
            t1a.switch_event(e1a, e2a)
            t1b.switch_event(e1b, e2b)
            schedule[e2a] = t1a
            schedule[e2b] = t1b
            if not is_match2_teams:
                del schedule[e1a]
                del schedule[e1b]

        if is_match2_teams:
            t2a.switch_event(e2a, e1a)
            t2b.switch_event(e2b, e1b)
            schedule[e1a] = t2a
            schedule[e1b] = t2b
            if not is_match1_teams:
                del schedule[e2a]
                del schedule[e2b]

        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the schedule schedule."""
        if not self.swap_candidates:
            return None

        n_candidates = len(self.swap_candidates)
        for idx in self.rng.permutation(n_candidates):
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data

            t1a, t1b = schedule[e1a], schedule[e1b]
            if None not in (t1a, t1b) and (t1a.conflicts(e2a, ignore=e1a) or t1b.conflicts(e2b, ignore=e1b)):
                continue

            t2a, t2b = schedule[e2a], schedule[e2b]
            if None not in (t2a, t2b) and (t2a.conflicts(e1a, ignore=e2a) or t2b.conflicts(e1b, ignore=e2b)):
                continue

            return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)

        return None, None


class SwapTableSideMutation(SwapMutation):
    """Mutation operator for swapping the sides of two tables in a match."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SWAP_TABLE_SIDE

    def mutate(self, schedule: Schedule) -> bool:
        """Swap the sides of two tables in a match."""
        match_data = self.get_swap_candidates(schedule)
        if match_data is None:
            return False

        e1a, e1b, t1a, t1b = match_data
        t1a.switch_event(e1a, e1b)
        t1b.switch_event(e1b, e1a)

        schedule[e1a] = t1b
        schedule[e1b] = t1a

        return True

    def get_swap_candidates(self, schedule: Schedule) -> Match | None:
        """Get two matches to swap in the schedule schedule."""
        if not self.swap_candidates:
            return None

        n_candidates = len(self.swap_candidates)
        for idx in self.rng.permutation(n_candidates):
            match_data, _ = self.swap_candidates[idx]
            e1, e2 = match_data

            t1, t2 = schedule[e1], schedule[e2]
            if None in (t1, t2):
                continue

            return e1, e2, t1, t2
        return None


@dataclass(slots=True)
class TimeSlotSequenceMutation(Mutation):
    """Abstract base class for mutations that permute assignments within a single timeslot."""

    event_factory: EventFactory
    event_map: dict[int, Event] = None
    timeslot_event_map: dict[tuple[RoundType, TimeSlot], list[Event]] = None
    timeslot_keys: list[tuple[RoundType, TimeSlot]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.timeslot_event_map = self.event_factory.as_timeslots()
        self.timeslot_keys = list(self.timeslot_event_map.keys())
        self.event_map = self.event_factory.as_mapping()

    @abstractmethod
    def permute(self, items: list) -> list:
        """Permute the list of items. To be implemented by subclasses."""
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def get_candidates(self, schedule: Schedule) -> list[Event]:
        """Get a list of candidate events for mutation within a specific timeslot."""
        chosen_key_i = self.rng.integers(0, len(self.timeslot_keys))
        chosen_key = self.timeslot_keys[chosen_key_i]
        chosen_rt, _ = chosen_key

        max_len_timeslot = max(len(v) for k, v in self.timeslot_event_map.items() if k[0] == chosen_rt)
        s_indices = schedule.scheduled_event_indices()
        schedule_events = (self.event_map[i] for i in s_indices)
        less_than_timeslots = [
            k
            for k, _ in self.timeslot_event_map.items()
            if k[0] == chosen_rt and len([e for e in schedule_events if e.timeslot == k[1]]) < max_len_timeslot
        ]

        if less_than_timeslots and self.rng.random() < 0.5:
            chosen_key_i = self.rng.integers(0, len(less_than_timeslots))
            chosen_key = less_than_timeslots[chosen_key_i]

        return sorted(self.timeslot_event_map[chosen_key], key=lambda e: e.location.idx)

    def mutate(self, schedule: Schedule) -> bool:
        """Find a suitable timeslot and round type, then permute assignments."""
        candidates = self.get_candidates(schedule)
        tpr = candidates[0].location.teams_per_round
        if tpr == 1:
            return self.mutate_singles(schedule, candidates)
        if tpr == 2:
            return self.mutate_matches(schedule, candidates)
        return False

    def mutate_singles(self, schedule: Schedule, candidates: list[Event]) -> bool:
        """Permute team assignments for single-team events."""
        old_ids = [None if schedule[e] is None else schedule[e].idx for e in candidates]
        new_ids = self.permute(old_ids)
        for event, old_id, new_id in zip(candidates, old_ids, new_ids, strict=True):
            if old_id == new_id:
                continue

            if old_id is not None and (old_team := schedule.get_team(old_id)):
                old_team.remove_event(event)
                del schedule[event]

            if new_id is not None and (new_team := schedule.get_team(new_id)):
                new_team.add_event(event)
                schedule[event] = new_team

        return True

    def mutate_matches(self, schedule: Schedule, candidates: list[Event]) -> bool:
        """Permute team assignments for match-based events."""
        matches: list[tuple[Event, ...]] = []
        teams: list[tuple[Team | None, ...]] = []
        for e1 in candidates:
            if e1.location.side == 1 and (e2 := e1.paired):
                t1, t2 = schedule[e1], schedule[e2]
                matches.append((e1, e2))
                teams.append((t1, t2))

        old_ids: list[tuple[int | None, ...]] = [
            (t1, t2) if None in (t1, t2) else (t1.idx, t2.idx) for (t1, t2) in teams
        ]
        new_ids = self.permute(old_ids)
        for match, old_id_pair, new_id_pair in zip(matches, old_ids, new_ids, strict=True):
            if old_id_pair == new_id_pair:
                continue

            e1, e2 = match

            if None in old_id_pair:
                old_t1, old_t2 = old_id_pair
            else:
                old_t1, old_t2 = (schedule.get_team(i) for i in old_id_pair)
                old_t1.remove_event(e1)
                old_t2.remove_event(e2)
                del schedule[e1]
                del schedule[e2]

            if None in new_id_pair:
                new_t1, new_t2 = new_id_pair
            else:
                new_t1, new_t2 = (schedule.get_team(i) for i in new_id_pair)
                new_t1.add_event(e1)
                new_t2.add_event(e2)
                schedule[e1] = new_t1
                schedule[e2] = new_t2

        return True


@dataclass(slots=True)
class InversionMutation(TimeSlotSequenceMutation):
    """Inverts a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.INVERSION

    def permute(
        self, items: list[int | None | tuple[list | None, ...]]
    ) -> Iterator[int | None | tuple[list | None, ...]]:
        """Invert a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)

        if isinstance(items[0], tuple):
            return reversed([tuple(reversed(pair)) for pair in items])

        return reversed(items[:])


@dataclass(slots=True)
class ScrambleMutation(TimeSlotSequenceMutation):
    """Scrambles a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SCRAMBLE

    def permute(
        self, items: list[int | None | tuple[list | None, ...]]
    ) -> Iterator[int | None | tuple[list | None, ...]]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)

        if isinstance(items[0], tuple):
            return (tuple(self.rng.permutation(pair)) for pair in items)

        return (_ for _ in self.rng.permutation(items))
