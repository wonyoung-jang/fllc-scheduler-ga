"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
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

    @abstractmethod
    def mutate(self, child: Schedule) -> None:
        """Mutate a child schedule to introduce genetic diversity."""

    def _validate_swap(self, event1: Event, event2: Event, *, same_timeslot: bool, same_location: bool) -> bool:
        """Check if the swap between two events is valid based on timeslot and location."""
        is_same_timeslot = event1.timeslot == event2.timeslot
        is_same_location = event1.location == event2.location
        if same_timeslot:
            return is_same_timeslot and not is_same_location
        if same_location:
            return is_same_location and not is_same_timeslot
        if not (same_timeslot or same_location):
            return not (is_same_timeslot or is_same_location)
        logger.warning("Invalid swap.")
        return False


@dataclass(slots=True, frozen=True)
class SwapTeamMutation(Mutation):
    """Mutation operator for swapping single team between two matches."""

    def _get_swap_candidates(
        self, child: Schedule, *, same_timeslot: bool, same_location: bool
    ) -> tuple[tuple[Event, Team, Team], tuple[Event, Team, Team]] | tuple[None, None]:
        """Get two matches to swap in the child schedule."""
        matches = child.get_matches()
        if len(matches) < 2:
            return None, None

        target_rtype = self.rng.choice(list(matches.keys()))
        event1_a, _, team1_a, team1_b = matches[target_rtype].pop()

        for event2_a, _, team2_a, team2_b in matches[target_rtype]:
            if self._validate_swap(event1_a, event2_a, same_timeslot=same_timeslot, same_location=same_location):
                if (
                    team1_a.identity in (team2_a.identity, team2_b.identity)
                    or team2_a.identity == team1_b.identity
                    or team1_a.conflicts(event2_a)
                    or team2_a.conflicts(event1_a)
                ):
                    continue

                return (event1_a, team1_a, team1_b), (event2_a, team2_a, team2_b)

        return None, None

    def mutate(self, child: Schedule, *, same_timeslot: bool = False, same_location: bool = False) -> None:
        """Swap one team from two different matches."""
        match1_data, match2_data = self._get_swap_candidates(
            child, same_timeslot=same_timeslot, same_location=same_location
        )

        if not match1_data:
            return

        event1_a, team1_a, team1_b = match1_data
        event2_a, team2_a, team2_b = match2_data

        team1_a.switch_event(event1_a, event2_a)
        team2_a.switch_event(event2_a, event1_a)

        team1_a.switch_opponent(team1_b, team2_b)
        team1_b.switch_opponent(team1_a, team2_a)
        team2_a.switch_opponent(team2_b, team1_b)
        team2_b.switch_opponent(team2_a, team1_a)

        child[event1_a], child[event2_a] = team2_a, team1_a


@dataclass(slots=True, frozen=True)
class SwapTeamWithinTimeSlot(SwapTeamMutation):
    """Mutation operator for swapping teams within the same time slot in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapTeamWithinTimeSlot, self).mutate(child, same_timeslot=True)


@dataclass(slots=True, frozen=True)
class SwapTeamWithinLocation(SwapTeamMutation):
    """Mutation operator for teams in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapTeamWithinLocation, self).mutate(child, same_location=True)


@dataclass(slots=True, frozen=True)
class SwapTeamAcrossLocation(SwapTeamMutation):
    """Mutation operator for teams in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapTeamAcrossLocation, self).mutate(child)


@dataclass(slots=True, frozen=True)
class SwapMatchMutation(Mutation):
    """Base class for mutations that swap the locations of two entire matches."""

    def _get_swap_candidates(
        self, child: Schedule, *, same_timeslot: bool, same_location: bool
    ) -> tuple[tuple[Event, Event, Team, Team], tuple[Event, Event, Team, Team]] | tuple[None, None]:
        """Get two matches to swap in the child schedule."""
        matches = child.get_matches()
        if len(matches) < 2:
            return None, None

        target_rtype = self.rng.choice(list(matches.keys()))
        event1_a, event1_b, team1_a, team1_b = matches[target_rtype].pop()

        for event2_a, event2_b, team2_a, team2_b in matches[target_rtype]:
            if self._validate_swap(event1_a, event2_a, same_timeslot=same_timeslot, same_location=same_location):
                if (
                    team1_a.conflicts(event2_a)
                    or team1_b.conflicts(event2_b)
                    or team2_a.conflicts(event1_a)
                    or team2_b.conflicts(event1_b)
                ):
                    continue

                return (event1_a, event1_b, team1_a, team1_b), (event2_a, event2_b, team2_a, team2_b)

        return None, None

    def mutate(self, child: Schedule, *, same_timeslot: bool = False, same_location: bool = False) -> None:
        """Swap two entire matches."""
        match1_data, match2_data = self._get_swap_candidates(
            child, same_timeslot=same_timeslot, same_location=same_location
        )

        if not match1_data:
            return

        event1_a, event1_b, team1_a, team1_b = match1_data
        event2_a, event2_b, team2_a, team2_b = match2_data

        team1_a.switch_event(event1_a, event2_a)
        team1_b.switch_event(event1_b, event2_b)
        team2_a.switch_event(event2_a, event1_a)
        team2_b.switch_event(event2_b, event1_b)

        child[event1_a], child[event1_b] = team2_a, team2_b
        child[event2_a], child[event2_b] = team1_a, team1_b


@dataclass(slots=True, frozen=True)
class SwapMatchWithinTimeSlot(SwapMatchMutation):
    """Mutation operator for swapping matches within the same time slot in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapMatchWithinTimeSlot, self).mutate(child, same_timeslot=True)


@dataclass(slots=True, frozen=True)
class SwapMatchWithinLocation(SwapMatchMutation):
    """Mutation operator for team matches in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapMatchWithinLocation, self).mutate(child, same_location=True)


@dataclass(slots=True, frozen=True)
class SwapMatchAcrossLocation(SwapMatchMutation):
    """Mutation operator for team matches in the FLL Scheduler GA."""

    def mutate(self, child: Schedule) -> None:
        """Perform a swap-team-within-time-slot mutation on a child schedule.

        Affects: OpponentVariety, TableConsistency
        """
        super(SwapMatchAcrossLocation, self).mutate(child)
