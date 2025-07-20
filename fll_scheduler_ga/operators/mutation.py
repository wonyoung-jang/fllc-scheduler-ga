"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..config.config import TournamentConfig
from ..data_model.event import Event
from ..genetic.schedule import Match, Schedule

logger = logging.getLogger(__name__)


def build_mutations(config: TournamentConfig, rng: Random) -> Iterator["Mutation"]:
    """Build and return a tuple of mutation operators based on the configuration."""
    if "genetic.mutation" not in config.parser:
        msg = "No mutation configuration section '[genetic.mutation]' found."
        raise ValueError(msg)

    variant_map = {
        # SwapMatchMutation variants
        "SwapMatch_CrossTimeLocation": lambda r: SwapMatchMutation(r, same_timeslot=False, same_location=False),
        "SwapMatch_SameLocation": lambda r: SwapMatchMutation(r, same_timeslot=False, same_location=True),
        "SwapMatch_SameTime": lambda r: SwapMatchMutation(r, same_timeslot=True, same_location=False),
        # SwapTeamMutation variants
        "SwapTeam_CrossTimeLocation": lambda r: SwapTeamMutation(r, same_timeslot=False, same_location=False),
        "SwapTeam_SameLocation": lambda r: SwapTeamMutation(r, same_timeslot=False, same_location=True),
        "SwapTeam_SameTime": lambda r: SwapTeamMutation(r, same_timeslot=True, same_location=False),
    }

    config_str = config.parser["genetic.mutation"].get("mutation_types", "")
    enabled_variants = [v.strip() for v in config_str.split(",") if v.strip()]

    if not enabled_variants:
        logger.warning("No mutation types enabled in the configuration. Mutation will not occur.")
        return

    for variant_name in enabled_variants:
        if variant_name not in variant_map:
            msg = f"Unknown mutation type in config: '{variant_name}'"
            raise ValueError(msg)
        else:
            mutation_factory = variant_map[variant_name]
            yield mutation_factory(rng)


@dataclass(slots=True, frozen=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: Random
    same_timeslot: bool = False
    same_location: bool = False

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

            event1a = match1_data[0]
            event2a = match2_data[0]

            if not self._validate_swap(event1a, event2a):
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

        if self.same_timeslot:
            return is_same_timeslot and not is_same_location

        if self.same_location:
            return is_same_location and not is_same_timeslot

        if not (self.same_timeslot or self.same_location):
            return not (is_same_timeslot or is_same_location)

        logger.warning("Invalid swap.")

        return False


@dataclass(slots=True, frozen=True)
class SwapTeamMutation(Mutation):
    """Mutation operator for swapping single team between two matches."""

    def get_swap_candidates(self, child: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the child schedule.

        Args:
            child (Schedule): The schedule to analyze.

        Returns:
            tuple[Match | None]: A tuple containing two matches to swap,
            or None if no valid candidates are found.

        """
        for match1_data, match2_data in self.yield_swap_candidates(child):
            event1a, _, team1a, team1b = match1_data
            event2a, _, team2a, team2b = match2_data

            match1_team_ids = {team1a.identity, team1b.identity}
            match2_team_ids = {team2a.identity, team2b.identity}

            if match1_team_ids.intersection(match2_team_ids):
                continue

            if team1a.conflicts(event2a) or team2a.conflicts(event1a):
                continue

            return match1_data, match2_data

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap one team from two different matches.

        Args:
            child (Schedule): The schedule to mutate.

        Returns:
            bool: True if the mutation was successful, False otherwise.

        """
        match1_data, match2_data = self.get_swap_candidates(child)

        if not match1_data:
            return False

        event1a, _, team1a, team1b = match1_data
        event2a, _, team2a, team2b = match2_data

        team1a.switch_opponent(team1b, team2b)
        team1b.switch_opponent(team1a, team2a)
        team2a.switch_opponent(team2b, team1b)
        team2b.switch_opponent(team2a, team1a)

        team1a.switch_event(event1a, event2a)
        team2a.switch_event(event2a, event1a)

        child[event1a] = team2a
        child[event2a] = team1a

        return True


@dataclass(slots=True, frozen=True)
class SwapMatchMutation(Mutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def get_swap_candidates(self, child: Schedule) -> tuple[Match | None]:
        """Get two matches to swap in the child schedule.

        Args:
            child (Schedule): The schedule to analyze.

        Returns:
            tuple[Match | None]: A tuple containing two matches to swap,

        """
        for match1_data, match2_data in self.yield_swap_candidates(child):
            event1a, event1b, team1a, team1b = match1_data
            event2a, event2b, team2a, team2b = match2_data

            if (
                team1a.conflicts(event2a)
                or team1b.conflicts(event2b)
                or team2a.conflicts(event1a)
                or team2b.conflicts(event1b)
            ):
                continue

            return match1_data, match2_data

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap two entire matches.

        Args:
            child (Schedule): The schedule to mutate.

        Returns:
            bool: True if the mutation was successful, False otherwise.

        """
        match1_data, match2_data = self.get_swap_candidates(child)

        if not match1_data:
            return False

        event1a, event1b, team1a, team1b = match1_data
        event2a, event2b, team2a, team2b = match2_data

        team1a.switch_event(event1a, event2a)
        team1b.switch_event(event1b, event2b)
        team2a.switch_event(event2a, event1a)
        team2b.switch_event(event2b, event1b)

        child[event1a] = team2a
        child[event1b] = team2b
        child[event2a] = team1a
        child[event2b] = team1b

        return True
