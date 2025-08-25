"""Genetic operators for FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from ..config.constants import MutationOp

if TYPE_CHECKING:
    from collections.abc import Iterator
    from random import Random

    from ..config.app_config import AppConfig
    from ..config.config import RoundType
    from ..data_model.schedule import Match, Schedule

logger = getLogger(__name__)


def build_mutations(app_config: AppConfig) -> Iterator[Mutation]:
    """Build and return a tuple of mutation operators based on the configuration."""
    variant_map = {
        # SwapMatchMutation variants
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda: SwapMatchMutation(
            rng=app_config.rng,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda: SwapMatchMutation(
            rng=app_config.rng,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda: SwapMatchMutation(
            rng=app_config.rng,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTeamMutation variants
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda: SwapTeamMutation(
            rng=app_config.rng,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda: SwapTeamMutation(
            rng=app_config.rng,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda: SwapTeamMutation(
            rng=app_config.rng,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTableSideMutation variant
        MutationOp.SWAP_TABLE_SIDE: lambda: SwapTableSideMutation(
            rng=app_config.rng,
            same_timeslot=True,
            same_location=True,
        ),
    }

    if not app_config.operators.mutation_types:
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return

    for variant_name in app_config.operators.mutation_types:
        if variant_name not in variant_map:
            msg = f"Unknown mutation type in config: '{variant_name}'"
            raise ValueError(msg)
        else:
            mutation_factory = variant_map[variant_name]
            yield mutation_factory()


@dataclass(slots=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: Random

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
class InversionMutation(Mutation):
    """Mutation operator for mutating an entire timeslot."""


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

    def yield_swap_candidates(self, schedule: Schedule) -> Iterator[tuple[Match, ...]]:
        """Yield candidates for swapping teams in matches.

        Args:
            schedule (Schedule): The schedule to analyze.

        Yields:
            tuple[Match, ...]: A tuple containing two matches to swap.

        """
        _same_timeslot = self.same_timeslot
        _same_location = self.same_location
        _match_pool = self._get_match_pool(schedule)
        for i, match1_data in enumerate(_match_pool, start=1):
            for match2_data in _match_pool[i:]:
                _e1, _, _, _ = match1_data
                _e2, _, _, _ = match2_data
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

    def _get_match_pool(self, schedule: Schedule) -> list[Match]:
        """Get a pool of matches from the schedule schedule."""
        _matches: dict[RoundType, list[Match]] = schedule.matches()

        if len(_matches) < 2:
            return []

        _roundtype = self.rng.choice(list(_matches.keys()))
        _match_pool = _matches[_roundtype]
        return self.rng.sample(_match_pool, k=len(_match_pool))

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
        for match1_data, match2_data in self.yield_swap_candidates(schedule):
            e1a, _, t1a, t1b = match1_data
            e2a, _, t2a, t2b = match2_data

            match_team_ids = {t1a.identity, t1b.identity, t2a.identity, t2b.identity}
            if len(match_team_ids) < 4 or t1a.conflicts(e2a, ignore=e1a) or t2a.conflicts(e1a, ignore=e2a):
                continue

            return match1_data, match2_data

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

        t1a.switch_event(e1a, e2a)
        t1b.switch_event(e1b, e2b)
        t2a.switch_event(e2a, e1a)
        t2b.switch_event(e2b, e1b)

        schedule[e1a] = t2a
        schedule[e1b] = t2b
        schedule[e2a] = t1a
        schedule[e2b] = t1b

        schedule.clear_cache()

        return True

    def get_swap_candidates(self, schedule: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the schedule schedule."""
        for match1_data, match2_data in self.yield_swap_candidates(schedule):
            e1a, e1b, t1a, t1b = match1_data
            e2a, e2b, t2a, t2b = match2_data

            if (
                t1a.conflicts(e2a, ignore=e1a)
                or t1b.conflicts(e2b, ignore=e1b)
                or t2a.conflicts(e1a, ignore=e2a)
                or t2b.conflicts(e1b, ignore=e2b)
            ):
                continue

            return match1_data, match2_data

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

        schedule.clear_cache()

        return True

    def get_swap_candidates(self, schedule: Schedule) -> Match | None:
        """Get two matches to swap in the schedule schedule."""
        next_swap = next(self.yield_swap_candidates(schedule), None)
        return self.rng.choice(next_swap) if next_swap is not None else None
