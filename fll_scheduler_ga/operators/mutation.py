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
    def get_swap_candidates(self, child: Schedule) -> tuple[Match | None]:
        """Get candidates for swapping teams in the child schedule.

        Args:
            child (Schedule): The schedule to analyze.

        Returns:
            tuple[Match | None]: A tuple containing two matches to swap,
            or None if no valid candidates are found.

        """
        msg = "get_swap_candidates method must be implemented by subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def mutate(self, child: Schedule) -> bool:
        """Mutate a child schedule to introduce genetic diversity.

        Args:
            child (Schedule): The schedule to mutate.

        Returns:
            bool: True if mutation was successful, False otherwise.

        """
        msg = "Mutate method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_match_pool(self, child: Schedule) -> Iterator[Match]:
        """Get a pool of matches from the child schedule.

        Args:
            child (Schedule): The schedule to analyze.

        Yields:
            Match: A match consisting of two events and their associated teams.

        """
        matches = child.get_matches()

        if len(matches) < 2:
            return

        target_roundtype = self.rng.choice(list(matches.keys()))
        match_pool = matches[target_roundtype]
        yield from self.rng.sample(match_pool, k=len(match_pool))

    def yield_swap_candidates(self, child: Schedule) -> Iterator[tuple[Match, ...]]:
        """Yield candidates for swapping teams in matches.

        Args:
            child (Schedule): The schedule to analyze.

        Yields:
            tuple[Match, ...]: A tuple containing two matches to swap.

        """
        match_pool = self.get_match_pool(child)

        for match1_data in match_pool:
            if (match2_data := next(match_pool, None)) is None:
                return

            e1a = match1_data[0]
            e2a = match2_data[0]

            if not self._validate_swap(e1a, e2a):
                continue

            yield match1_data, match2_data

    def _validate_swap(self, event1: Event, event2: Event) -> bool:
        """Check if the swap between two events is valid based on timeslot and location.

        Args:
            event1 (Event): The first event to swap.
            event2 (Event): The second event to swap.

        Returns:
            bool: True if the swap is valid, False otherwise.

        """
        is_same_timeslot = event1.timeslot == event2.timeslot
        is_same_location = event1.location == event2.location

        if self.same_timeslot and not self.same_location:
            return is_same_timeslot and not is_same_location

        if self.same_location and not self.same_timeslot:
            return is_same_location and not is_same_timeslot

        if not self.same_timeslot and not self.same_location:
            return not is_same_timeslot and not is_same_location

        if self.same_timeslot and self.same_location:
            return is_same_timeslot and is_same_location

        return False


@dataclass(slots=True)
class SwapTeamMutation(Mutation):
    """Mutation operator for swapping single team between two matches."""

    def get_swap_candidates(self, child: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the child schedule."""
        for match1_data, match2_data in self.yield_swap_candidates(child):
            e1a, _, t1a, t1b = match1_data
            e2a, _, t2a, t2b = match2_data

            match_team_ids = {t1a.identity, t1b.identity, t2a.identity, t2b.identity}

            if len(match_team_ids) < 4 or t1a.conflicts(e2a) or t2a.conflicts(e1a):
                continue

            return match1_data, match2_data

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap one team from two different matches."""
        match1_data, match2_data = self.get_swap_candidates(child)

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

        child[e1a] = t2a
        child[e2a] = t1a

        return True


@dataclass(slots=True)
class SwapMatchMutation(Mutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def get_swap_candidates(self, child: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the child schedule."""
        for match1_data, match2_data in self.yield_swap_candidates(child):
            e1a, e1b, t1a, t1b = match1_data
            e2a, e2b, t2a, t2b = match2_data

            if t1a.conflicts(e2a) or t1b.conflicts(e2b) or t2a.conflicts(e1a) or t2b.conflicts(e1b):
                continue

            return match1_data, match2_data

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap two entire matches."""
        match1_data, match2_data = self.get_swap_candidates(child)

        if not match1_data:
            return False

        e1a, e1b, t1a, t1b = match1_data
        e2a, e2b, t2a, t2b = match2_data

        t1a.switch_event(e1a, e2a)
        t1b.switch_event(e1b, e2b)
        t2a.switch_event(e2a, e1a)
        t2b.switch_event(e2b, e1b)

        child[e1a] = t2a
        child[e1b] = t2b
        child[e2a] = t1a
        child[e2b] = t1b

        return True


class SwapTableSideMutation(Mutation):
    """Mutation operator for swapping the sides of two tables in a match."""

    def get_swap_candidates(self, child: Schedule) -> Match | None:
        """Get two matches to swap in the child schedule."""
        return self.rng.choice(next(self.yield_swap_candidates(child), (None, None)))

    def mutate(self, child: Schedule) -> bool:
        """Swap the sides of two tables in a match."""
        match_data = self.get_swap_candidates(child)

        if not match_data:
            return False

        e1a, e1b, t1a, t1b = match_data
        t1a.switch_event(e1a, e1b)
        t1b.switch_event(e1b, e1a)
        child[e1a] = t1b
        child[e1b] = t1a
        return True
