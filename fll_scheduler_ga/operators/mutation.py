"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..config.app_config import AppConfig
from ..config.constants import MutationOp
from ..data_model.event import Event
from ..data_model.schedule import Match, Schedule

logger = logging.getLogger(__name__)


def build_mutations(app_config: AppConfig) -> Iterator["Mutation"]:
    """Build and return a tuple of mutation operators based on the configuration."""
    variant_map = {
        # SwapMatchMutation variants
        MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION: lambda: SwapMatchMutation(
            app_config.rng,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_MATCH_SAME_LOCATION: lambda: SwapMatchMutation(
            app_config.rng,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_MATCH_SAME_TIME: lambda: SwapMatchMutation(
            app_config.rng,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTeamMutation variants
        MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION: lambda: SwapTeamMutation(
            app_config.rng,
            same_timeslot=False,
            same_location=False,
        ),
        MutationOp.SWAP_TEAM_SAME_LOCATION: lambda: SwapTeamMutation(
            app_config.rng,
            same_timeslot=False,
            same_location=True,
        ),
        MutationOp.SWAP_TEAM_SAME_TIME: lambda: SwapTeamMutation(
            app_config.rng,
            same_timeslot=True,
            same_location=False,
        ),
        # SwapTableSideMutation variant
        MutationOp.SWAP_TABLE_SIDE: lambda: SwapTableSideMutation(
            app_config.rng,
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

    def get_match_pool(self, schedule: Schedule) -> list[Match]:
        """Get a pool of matches from the schedule schedule.

        Args:
            schedule (Schedule): The schedule to analyze

        Returns:
            list[Match]: A list of matches consisting of two events and their associated teams.

        """
        matches = schedule.matches()

        if len(matches) < 2:
            return []

        match_pool = matches[self.rng.choice(list(matches.keys()))]
        return self.rng.sample(match_pool, k=len(match_pool))

    def yield_swap_candidates(self, schedule: Schedule) -> Iterator[tuple[Match, ...]]:
        """Yield candidates for swapping teams in matches.

        Args:
            schedule (Schedule): The schedule to analyze.

        Yields:
            tuple[Match, ...]: A tuple containing two matches to swap.

        """
        match_pool = self.get_match_pool(schedule)
        for i, match1_data in enumerate(match_pool, start=1):
            for match2_data in match_pool[i:]:
                e1a, _, _, _ = match1_data
                e2a, _, _, _ = match2_data

                if not self._validate_swap(e1a, e2a):
                    continue

                yield match1_data, match2_data

    def _validate_swap(self, e1: Event, e2: Event) -> bool:
        """Check if the swap between two events is valid based on timeslot and location.

        Args:
            e1 (Event): The first event to swap.
            e2 (Event): The second event to swap.

        Returns:
            bool: True if the swap is valid, False otherwise.

        """
        is_same_timeslot = e1.timeslot == e2.timeslot
        is_same_location = e1.location == e2.location

        if self.same_timeslot and not self.same_location:
            return is_same_timeslot and not is_same_location

        if self.same_location and not self.same_timeslot:
            return is_same_location and not is_same_timeslot

        if not self.same_timeslot and not self.same_location:
            return not is_same_timeslot and not is_same_location

        return self.same_timeslot and self.same_location


@dataclass(slots=True)
class SwapTeamMutation(Mutation):
    """Mutation operator for swapping single team between two matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_TEAM_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_TEAM_SAME_LOCATION,
            (False, True): MutationOp.SWAP_TEAM_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

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

    def mutate(self, schedule: Schedule) -> bool:
        """Swap one team from two different matches."""
        match1_data, match2_data = self.get_swap_candidates(schedule)

        if not match1_data:
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


@dataclass(slots=True)
class SwapMatchMutation(Mutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def __str__(self) -> str:
        """Return string representation."""
        return {
            (False, False): MutationOp.SWAP_MATCH_CROSS_TIME_LOCATION,
            (True, False): MutationOp.SWAP_MATCH_SAME_LOCATION,
            (False, True): MutationOp.SWAP_MATCH_SAME_TIME,
        }.get((self.same_location, self.same_timeslot), self.__class__.__name__)

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

    def mutate(self, schedule: Schedule) -> bool:
        """Swap two entire matches."""
        match1_data, match2_data = self.get_swap_candidates(schedule)

        if not match1_data:
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


class SwapTableSideMutation(Mutation):
    """Mutation operator for swapping the sides of two tables in a match."""

    def __str__(self) -> str:
        """Return string representation."""
        return MutationOp.SWAP_TABLE_SIDE

    def get_swap_candidates(self, schedule: Schedule) -> Match | None:
        """Get two matches to swap in the schedule schedule."""
        next_swap = next(self.yield_swap_candidates(schedule), None)
        return self.rng.choice(next_swap) if next_swap is not None else None

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
