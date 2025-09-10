"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import MutationOp

if TYPE_CHECKING:
    from collections.abc import Iterator
    from random import Random

    from ..config.app_config import AppConfig
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

    rng: Random
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

    def yield_swap_candidates(self) -> Iterator[tuple[Match, ...]]:
        """Yield candidates for swapping teams in matches.

        Yields:
            tuple[Match, ...]: A tuple containing two matches to swap.

        """
        _same_timeslot = self.same_timeslot
        _same_location = self.same_location
        _match_pool = self._get_match_pool()
        for match1_data, match2_data in combinations(_match_pool, 2):
            _e1, _ = match1_data
            _e2, _ = match2_data
            _is_same_timeslot = _e1.timeslot == _e2.timeslot
            _is_same_location = _e1.location == _e2.location

            if not self._validate_swap(
                is_same_timeslot=_is_same_timeslot,
                is_same_location=_is_same_location,
                same_timeslot=_same_timeslot,
                same_location=_same_location,
            ):
                continue

            yield match1_data, match2_data

    def _get_match_pool(self) -> list[tuple[Event, ...]]:
        """Get a pool of matches from the schedule."""
        _matches = self.event_factory.as_matches()
        if len(_matches) < 2:
            return []

        _roundtype = self.rng.choice(list(_matches.keys()))
        return self.rng.sample(_matches[_roundtype], k=len(_matches[_roundtype]))

    def _validate_swap(
        self,
        *,
        is_same_timeslot: bool,
        is_same_location: bool,
        same_timeslot: bool,
        same_location: bool,
    ) -> bool:
        """Check if the swap between two events is valid based on timeslot and location."""
        if same_timeslot and not same_location:
            return is_same_timeslot and not is_same_location

        if same_location and not same_timeslot:
            return is_same_location and not is_same_timeslot

        if not same_timeslot and not same_location:
            return not is_same_timeslot and not is_same_location

        return same_timeslot and same_location


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

        e1a, _, t1a, t1b = match1_data
        e2a, _, t2a, t2b = match2_data

        t1a.switch_opponent(t1b, t2b)
        t1b.switch_opponent(t1a, t2a)
        t2a.switch_opponent(t2b, t1b)
        t2b.switch_opponent(t2a, t1a)

        t1a.switch_event(e1a, e2a)
        t2a.switch_event(e2a, e1a)

        schedule[e1a] = t2a
        schedule[e2a] = t1a

        schedule.clear_cache()

        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the schedule schedule."""
        for match1_data, match2_data in self.yield_swap_candidates():
            e1a, e1b = match1_data
            e2a, e2b = match2_data
            t1a, t1b = schedule[e1a], schedule[e1b]
            t2a, t2b = schedule[e2a], schedule[e2b]
            if None in (t1a, t1b, t2a, t2b):
                continue

            match_team_ids = {t1a.identity, t1b.identity, t2a.identity, t2b.identity}
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

        schedule.clear_cache()
        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the schedule schedule."""
        for match1_data, match2_data in self.yield_swap_candidates():
            e1a, e1b = match1_data
            e2a, e2b = match2_data

            t1a, t1b = schedule[e1a], schedule[e1b]

            if None in (t1a, t1b):
                continue

            if t1a.conflicts(e2a, ignore=e1a) or t1b.conflicts(e2b, ignore=e1b):
                continue

            t2a, t2b = schedule[e2a], schedule[e2b]

            if None in (t2a, t2b):
                continue

            if t2a.conflicts(e1a, ignore=e2a) or t2b.conflicts(e1b, ignore=e2b):
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
        e1a, e1b = None, None
        match_pool = self.get_swap_candidates()
        for e1, e2 in match_pool:
            if None not in (schedule[e1], schedule[e2]):
                e1a, e1b = (e1, e2)
                break
        else:
            return False

        t1a, t1b = schedule[e1a], schedule[e1b]

        if None in (t1a, t1b):
            return False

        t1a.switch_event(e1a, e1b)
        t1b.switch_event(e1b, e1a)

        schedule[e1a] = t1b
        schedule[e1b] = t1a

        schedule.clear_cache()

        return True

    def get_swap_candidates(self) -> Match | None:
        """Get two matches to swap in the schedule schedule."""
        return self._get_match_pool()


@dataclass(slots=True)
class TimeSlotSequenceMutation(Mutation):
    """Abstract base class for mutations that permute assignments within a single timeslot."""

    event_factory: EventFactory
    _timeslot_event_map: dict[TimeSlot, list[Event]] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self._timeslot_event_map = self.event_factory.as_timeslots()

    @abstractmethod
    def _permute(self, items: list) -> list:
        """Permute the list of items. To be implemented by subclasses."""
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def mutate(self, schedule: Schedule) -> bool:
        """Find a suitable timeslot and round type, then permute assignments."""
        chosen_key = self.rng.choice(list(self._timeslot_event_map.keys()))
        chosen_rt, _ = chosen_key
        max_len_timeslot = max(len(v) for k, v in self._timeslot_event_map.items() if k[0] == chosen_rt)
        schedule_events = schedule.keys()
        less_than_timeslots = [
            k
            for k, _ in self._timeslot_event_map.items()
            if k[0] == chosen_rt and len([e for e in schedule_events if e.timeslot == k[1]]) < max_len_timeslot
        ]

        if less_than_timeslots and self.rng.choice((True, False)):
            chosen_rt_ts = self.rng.choice(less_than_timeslots)
        else:
            chosen_rt_ts = chosen_key

        candidates = self._timeslot_event_map[chosen_rt_ts]
        tpr = candidates[0].location.teams_per_round
        mutate_map = {
            1: self._mutate_singles,
            2: self._mutate_matches,
        }
        mutate_fn = mutate_map.get(tpr)
        if mutate_fn:
            return mutate_fn(schedule, candidates)

        return False

    def _mutate_singles(self, schedule: Schedule, candidates: list[Event]) -> bool:
        """Permute team assignments for single-team events."""
        candidates.sort(key=lambda e: (e.location.identity, e.location.side))
        original_ids = [None if schedule[e] is None else schedule[e].identity for e in candidates]
        permuted_ids = self._permute(original_ids)
        for event, old_id, new_id in zip(candidates, original_ids, permuted_ids, strict=True):
            if old_id == new_id:
                continue

            old_team = schedule.get_team(old_id)
            new_team = schedule.get_team(new_id)

            if old_team is not None:
                old_team.remove_event(event)
                del schedule[event]

            if new_team is not None:
                new_team.add_event(event)
                schedule[event] = new_team

        schedule.clear_cache()
        return True

    def _mutate_matches(self, schedule: Schedule, candidates: list[Event]) -> bool:
        """Permute team assignments for match-based events."""
        event_pairs: list[tuple[Event, ...]] = []
        teams: list[tuple[Team | None, ...]] = []
        candidates.sort(key=lambda e: (e.location.identity, e.location.side))

        for e1 in candidates:
            if e1.location.side == 1 and (e2 := e1.paired):
                t1, t2 = schedule[e1], schedule[e2]
                event_pairs.append((e1, e2))
                teams.append((t1, t2))

        original_ids = [(None, None) if None in (t1, t2) else (t1.identity, t2.identity) for (t1, t2) in teams]
        permuted_ids = self._permute(original_ids)
        for event_pair, old_ids, new_ids in zip(event_pairs, original_ids, permuted_ids, strict=True):
            if old_ids == new_ids:
                continue

            e1, e2 = event_pair
            old_t1, old_t2 = (schedule.get_team(i) for i in old_ids)
            new_t1, new_t2 = (schedule.get_team(i) for i in new_ids)

            if None not in (old_t1, old_t2):
                old_t1.remove_event(e1)
                old_t1.remove_opponent(old_t2)
                old_t2.remove_event(e2)
                old_t2.remove_opponent(old_t1)
                del schedule[e1]
                del schedule[e2]

            if None not in (new_t1, new_t2):
                new_t1.add_event(e1)
                new_t1.add_opponent(new_t2)
                new_t2.add_event(e2)
                new_t2.add_opponent(new_t1)
                schedule[e1] = new_t1
                schedule[e2] = new_t2

        schedule.clear_cache()
        return True


@dataclass(slots=True)
class InversionMutation(TimeSlotSequenceMutation):
    """Inverts a sub-sequence of assignments within a single timeslot."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.INVERSION

    def _permute(
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

    def _permute(
        self, items: list[int | None | tuple[list | None, ...]]
    ) -> Iterator[int | None | tuple[list | None, ...]]:
        """Scramble a random sub-sequence of the items."""
        if len(items) <= 1:
            return iter(items)

        if isinstance(items[0], tuple):
            return (tuple(self.rng.sample(pair, 2)) for pair in items)

        return (_ for _ in self.rng.sample(items[:], len(items)))
