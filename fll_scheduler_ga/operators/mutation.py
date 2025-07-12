"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from ..data_model.event import Event
from ..data_model.team import Team
from ..genetic.schedule import Schedule

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: Random
    same_timeslot: bool = False
    same_location: bool = False

    @abstractmethod
    def mutate(self, child: Schedule) -> bool:
        """Mutate a child schedule to introduce genetic diversity."""

    def get_match_pool(self, child: Schedule) -> Iterator[tuple[Event, Event, Team, Team]]:
        """Get a pool of matches from the child schedule."""
        matches = child.get_matches()

        if len(matches) < 2:
            return

        match_pool = matches[self.rng.choice(list(matches.keys()))]
        yield from self.rng.sample(match_pool, k=len(match_pool))

    def _validate_swap(self, event1: Event, event2: Event) -> bool:
        """Check if the swap between two events is valid based on timeslot and location."""
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

    def _get_swap_candidates(self, child: Schedule) -> tuple[tuple[Event, Team, Team] | None]:
        """Get two matches to swap in the child schedule."""
        match_pool = list(self.get_match_pool(child))

        for i, match1_data in enumerate(match_pool):
            event1_a, _, team1_a, team1_b = match1_data
            next_i = i + 1

            for match2_data in match_pool[next_i:]:
                event2_a, _, team2_a, team2_b = match2_data

                if not self._validate_swap(event1_a, event2_a):
                    continue

                if team1_a.identity in (team2_a.identity, team2_b.identity) or team2_a.identity == team1_b.identity:
                    continue

                if team1_a.conflicts(event2_a) or team2_a.conflicts(event1_a):
                    continue

                return (event1_a, team1_a, team1_b), (event2_a, team2_a, team2_b)

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap one team from two different matches."""
        match1_data, match2_data = self._get_swap_candidates(child)

        if not match1_data:
            return False

        event1_a, team1_a, team1_b = match1_data
        event2_a, team2_a, team2_b = match2_data

        team1_a.switch_opponent(team1_b, team2_b)
        team1_b.switch_opponent(team1_a, team2_a)
        team2_a.switch_opponent(team2_b, team1_b)
        team2_b.switch_opponent(team2_a, team1_a)

        del child[event1_a]
        del child[event2_a]
        child[event1_a] = team2_a
        child[event2_a] = team1_a

        return True


@dataclass(slots=True, frozen=True)
class SwapMatchMutation(Mutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def _get_swap_candidates(self, child: Schedule) -> tuple[tuple[Event, Event, Team, Team] | None]:
        """Get two matches to swap in the child schedule."""
        match_pool = list(self.get_match_pool(child))

        for i, match1_data in enumerate(match_pool):
            event1_a, event1_b, team1_a, team1_b = match1_data
            next_i = i + 1

            for match2_data in match_pool[next_i:]:
                event2_a, event2_b, team2_a, team2_b = match2_data

                if not self._validate_swap(event1_a, event2_a):
                    continue

                if (
                    team1_a.conflicts(event2_a)
                    or team1_b.conflicts(event2_b)
                    or team2_a.conflicts(event1_a)
                    or team2_b.conflicts(event1_b)
                ):
                    continue

                return (event1_a, event1_b, team1_a, team1_b), (event2_a, event2_b, team2_a, team2_b)

        return None, None

    def mutate(self, child: Schedule) -> bool:
        """Swap two entire matches."""
        match1_data, match2_data = self._get_swap_candidates(child)

        if not match1_data:
            return False

        event1_a, event1_b, team1_a, team1_b = match1_data
        event2_a, event2_b, team2_a, team2_b = match2_data

        del child[event1_a]
        del child[event1_b]
        del child[event2_a]
        del child[event2_b]
        child[event1_a] = team2_a
        child[event1_b] = team2_b
        child[event2_a] = team1_a
        child[event2_b] = team1_b

        return True
