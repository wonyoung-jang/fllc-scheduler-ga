"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import MutationOp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config.schemas import OperatorConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.schedule import Schedule

type Match = tuple[int, int, int, int]

logger = getLogger(__name__)


def build_mutations(
    rng: np.random.Generator,
    operators: OperatorConfig,
    event_factory: EventFactory,
    event_properties: EventProperties,
) -> tuple[Mutation, ...]:
    """Build and return a tuple of mutation operators based on the configuration."""
    if not (mutation_types := operators.mutation.types):
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return ()

    mutations = []
    mutation_factory = {
        # SwapMatchMutation variants
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda p: SwapMatchMutation(
            **p,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda p: SwapMatchMutation(
            **p,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda p: SwapMatchMutation(
            **p,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTeamMutation variants
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda p: SwapTeamMutation(
            **p,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda p: SwapTeamMutation(
            **p,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda p: SwapTeamMutation(
            **p,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTableSideMutation variant
        MutationOp.SWAP_TABLE_SIDE: lambda p: SwapTableSideMutation(
            **p,
            same_timeslot=True,
            same_location=True,
        ),
        # TimeSlotSequenceMutation variants
        MutationOp.INVERSION: lambda p: InversionMutation(**p),
        MutationOp.SCRAMBLE: lambda p: ScrambleMutation(**p),
    }
    params = {
        "rng": rng,
        "event_factory": event_factory,
        "event_properties": event_properties,
    }

    for mutation_name in mutation_types:
        if mutation_name not in mutation_factory:
            msg = f"Unknown mutation type in config: '{mutation_name}'"
            raise ValueError(msg)

        mutations.append(mutation_factory[mutation_name](params))

    return tuple(mutations)


@dataclass(slots=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: np.random.Generator
    event_factory: EventFactory
    event_properties: EventProperties

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

    same_timeslot: bool
    same_location: bool

    swap_candidates: list[tuple[tuple[int, ...], ...]] = field(default_factory=list)
    n_swap_candidates: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.swap_candidates.extend(self.init_swap_candidates())
        self.n_swap_candidates = len(self.swap_candidates)
        logger.debug("Initialized %d swap candidates for %s", self.n_swap_candidates, str(self))

    @abstractmethod
    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get candidates for swapping teams in the schedule schedule.

        Args:
            schedule (Schedule): The schedule to analyze.

        Returns:
            tuple[Match, ...] | tuple[None, ...]: The matches selected for swapping, or None if no valid swap found.

        """
        msg = "get_swap_candidates method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def init_swap_candidates(self) -> Iterator[tuple[tuple[int, ...], ...]]:
        """Precompute any necessary data before mutation."""
        _ts_idx = self.event_properties.timeslot_idx
        _loc_idx = self.event_properties.loc_idx
        _as_matches = self.event_factory.as_matches()

        for match_list in _as_matches.values():
            for match1, match2 in itertools.combinations(match_list, 2):
                e1a, e1b = match1
                e2a, e2b = match2
                is_same_timeslot = _ts_idx[e1a] == _ts_idx[e2a]
                is_same_location = _loc_idx[e1a] == _loc_idx[e2a]

                if not (self.same_timeslot and self.same_location) and not self._validate_swap(
                    is_same_timeslot=is_same_timeslot,
                    is_same_location=is_same_location,
                    same_timeslot=self.same_timeslot,
                    same_location=self.same_location,
                ):
                    continue

                match1_idx = (e1a, e1b)
                match2_idx = (e2a, e2b)
                yield (match1_idx, match2_idx)

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
        if self.n_swap_candidates <= 0:
            return False

        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False

        e1a, _, t1a, _ = match1_data
        e2a, _, t2a, _ = match2_data
        schedule.swap_assignment(t1a, e1a, e2a)
        schedule.swap_assignment(t2a, e2a, e1a)
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        shuffled_idx = self.rng.permutation(self.n_swap_candidates)
        for idx in shuffled_idx:
            idx: int
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data
            t1a, t1b = schedule[e1a], schedule[e1b]
            t2a, t2b = schedule[e2a], schedule[e2b]
            match_team_ids = {t1a, t1b, t2a, t2b}
            if (
                len(match_team_ids) < 4
                or schedule.conflicts(t1a, e2a, ignore=e1a)
                or schedule.conflicts(t2a, e1a, ignore=e2a)
            ):
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
        if self.n_swap_candidates <= 0:
            return False

        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False

        e1a, e1b, t1a, t1b = match1_data
        e2a, e2b, t2a, t2b = match2_data
        none_in_m1 = -1 in (t1a, t1b)
        none_in_m2 = -1 in (t2a, t2b)

        if not none_in_m1:
            schedule.swap_assignment(t1a, e1a, e2a)
            schedule.swap_assignment(t1b, e1b, e2b)

        if not none_in_m2:
            schedule.swap_assignment(t2a, e2a, e1a)
            schedule.swap_assignment(t2b, e2b, e1b)

        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get two matches to swap in the schedule schedule."""
        shuffled_idx = self.rng.permutation(self.n_swap_candidates)
        for idx in shuffled_idx:
            idx: int
            match1_data, match2_data = self.swap_candidates[idx]
            e1a, e1b = match1_data
            e2a, e2b = match2_data

            t1a, t1b = schedule[e1a], schedule[e1b]
            if -1 not in (t1a, t1b) and (
                schedule.conflicts(t1a, e2a, ignore=e1a) or schedule.conflicts(t1b, e2b, ignore=e1b)
            ):
                continue

            t2a, t2b = schedule[e2a], schedule[e2b]
            if -1 not in (t2a, t2b) and (
                schedule.conflicts(t2a, e1a, ignore=e2a) or schedule.conflicts(t2b, e1b, ignore=e2b)
            ):
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
        if self.n_swap_candidates <= 0:
            return False

        match1_data, match2_data = self.get_swap_candidates(schedule)
        if match1_data is None or match2_data is None:
            return False

        e1a, e1b, t1a, t1b = match1_data
        schedule.swap_assignment(t1a, e1a, e1b)
        schedule.swap_assignment(t1b, e1b, e1a)
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match, ...] | tuple[None, ...]:
        """Get one match to swap sides in the schedule schedule."""
        idx = self.rng.integers(0, self.n_swap_candidates)
        match1_data, match2_data = self.swap_candidates[idx]
        e1a, e1b = match1_data
        e2a, e2b = match2_data
        t1a, t1b = schedule[e1a], schedule[e1b]
        t2a, t2b = schedule[e2a], schedule[e2b]
        return (e1a, e1b, t1a, t1b), (e2a, e2b, t2a, t2b)


@dataclass(slots=True)
class TimeSlotSequenceMutation(Mutation):
    """Abstract base class for mutations that permute assignments within a single timeslot."""

    timeslot_candidates: dict[tuple[int, int], list[tuple[int, ...]]] = field(init=False)
    timeslot_keys: tuple[tuple[int, int], ...] = field(init=False)
    key_to_tpr: dict[tuple[int, int], int] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.timeslot_candidates, self.key_to_tpr = self.init_candidates()
        self.timeslot_keys = tuple(self.timeslot_candidates.keys())

    @abstractmethod
    def permute_singles(self, items: list[int]) -> Iterator[int]:
        """Permute the list of items. To be implemented by subclasses."""

    @abstractmethod
    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Permute the list of items. To be implemented by subclasses."""

    def init_candidates(
        self,
    ) -> tuple[dict[tuple[int, int], list[tuple[int, ...]]], dict[tuple[int, int], int]]:
        """Precompute candidate events for each timeslot."""
        ep = self.event_properties
        timeslot_data = {}
        keys_to_tpr = {}
        timeslot_event_map = self.event_factory.as_timeslots()
        for key, events in timeslot_event_map.items():
            candidates = [e for e in events if ep.loc_side[e] == 1 or ep.paired_idx[e] == -1]
            timeslot_data[key] = [(e, ep.paired_idx[e]) for e in candidates]
            keys_to_tpr[key] = ep.teams_per_round[events[0]]
        return timeslot_data, keys_to_tpr

    def get_candidates(self) -> tuple[list[tuple[int, ...]], int]:
        """Get a list of candidate events for mutation within a specific timeslot."""
        indices = np.arange(len(self.timeslot_keys))
        self.rng.shuffle(indices)
        idx = indices[0]
        key = self.timeslot_keys[idx]
        candidates = self.timeslot_candidates[key]
        tpr = self.key_to_tpr[key]
        return candidates, tpr

    def mutate(self, schedule: Schedule) -> bool:
        """Find a suitable timeslot and round type, then permute assignments."""
        candidates, tpr = self.get_candidates()
        if tpr == 1:
            return self.mutate_singles(schedule, candidates)
        if tpr == 2:
            return self.mutate_matches(schedule, candidates)
        return False

    def mutate_singles(self, schedule: Schedule, candidates: list[tuple[int, ...]]) -> bool:
        """Permute team assignments for single-team events."""
        old_ids = [schedule[e] for e, _ in candidates]
        new_ids = self.permute_singles(old_ids)
        for (event, _), old_team, new_team in zip(candidates, old_ids, new_ids, strict=True):
            if old_team == new_team:
                continue

            schedule.unassign(old_team, event)
            schedule.assign(new_team, event)

        return True

    def mutate_matches(self, schedule: Schedule, candidates: list[tuple[int, ...]]) -> bool:
        """Permute team assignments for match-based events."""
        matches: list[tuple[int, ...]] = []
        old_ids: list[tuple[int, ...]] = []
        for e1, e2 in candidates:
            t1, t2 = schedule[e1], schedule[e2]
            matches.append((e1, e2))
            old_ids.append((t1, t2))

        new_ids = self.permute_matches(old_ids)
        for (e1, e2), old_id_pair, new_id_pair in zip(matches, old_ids, new_ids, strict=True):
            if old_id_pair == new_id_pair:
                continue

            old_t1, old_t2 = old_id_pair
            schedule.unassign(old_t1, e1)
            schedule.unassign(old_t2, e2)

            new_t1, new_t2 = new_id_pair
            schedule.assign(new_t1, e1)
            schedule.assign(new_t2, e2)

        return True


@dataclass(slots=True)
class InversionMutation(TimeSlotSequenceMutation):
    """Inverts a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.INVERSION

    def permute_singles(self, items: list[int]) -> Iterator[int]:
        """Invert a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return reversed(items[:])

    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Invert a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return reversed([tuple(reversed(pair)) for pair in items])


@dataclass(slots=True)
class ScrambleMutation(TimeSlotSequenceMutation):
    """Scrambles a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SCRAMBLE

    def permute_singles(self, items: list[int]) -> Iterator[int]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return iter(self.rng.permutation(items))

    def permute_matches(self, items: list[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)
        return (tuple(self.rng.permutation(pair)) for pair in items)
