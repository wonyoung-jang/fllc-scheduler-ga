"""Genetic operators for FLL Scheduler GA."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random

from ..data_model.event import Event
from ..data_model.team import Team
from ..genetic.schedule import Schedule

logger = logging.getLogger(__name__)

type TableEventAndTeams = tuple[Event, Event, Team, Team]


def _get_scheduled_match_events(child: Schedule) -> list[TableEventAndTeams]:
    """Get a list of tuples representing all scheduled matches.

    Each tuple is (primary_event, opponent_event, team1, team2).
    """
    matches = []
    for event, team1 in child.items():
        if not event.paired_event or event.location.side != 1:
            continue

        if team2 := child[event.paired_event]:
            matches.append((event, event.paired_event, team1, team2))

    return matches


@dataclass(slots=True, frozen=True)
class Mutation(ABC):
    """Abstract base class for mutation operators in the FLL Scheduler GA."""

    rng: Random

    @abstractmethod
    def mutate(self, child: Schedule) -> None:
        """Mutate a child schedule to introduce genetic diversity."""


@dataclass(slots=True, frozen=True)
class SwapTeamMutation(Mutation):
    """Mutation operator for swapping single team between two matches."""

    def _get_swap_candidates(self, child: Schedule, *, same_timeslot: bool, same_location: bool) -> tuple:
        """Get two matches to swap in the child schedule."""
        matches = _get_scheduled_match_events(child)
        if len(matches) < 2:
            return None, None

        self.rng.shuffle(matches)
        event1_a, _, team1_a, team1_b = matches.pop()
        target_rtype = event1_a.round_type

        for event2_a, _, team2_a, team2_b in matches:
            if event2_a.round_type != target_rtype:
                continue

            if (
                (
                    same_timeslot
                    and event1_a.time_slot == event2_a.time_slot
                    and event1_a.location.identity != event2_a.location.identity
                )
                or (
                    same_location
                    and event1_a.location.identity == event2_a.location.identity
                    and event1_a.location.side == event2_a.location.side
                    and event1_a.time_slot != event2_a.time_slot
                )
                or (
                    not same_timeslot
                    and not same_location
                    and event1_a.time_slot != event2_a.time_slot
                    and event1_a.location.identity != event2_a.location.identity
                )
            ):
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
            child,
            same_timeslot=same_timeslot,
            same_location=same_location,
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

    def _get_swap_candidates(self, child: Schedule, *, same_timeslot: bool, same_location: bool) -> tuple:
        """Get two matches to swap in the child schedule."""
        matches = _get_scheduled_match_events(child)
        if len(matches) < 2:
            return None, None

        self.rng.shuffle(matches)
        event1_a, event1_b, team1_a, team1_b = matches.pop()
        target_rtype = event1_a.round_type

        for event2_a, event2_b, team2_a, team2_b in matches:
            if event2_a.round_type != target_rtype:
                continue

            if (
                (
                    same_timeslot
                    and team1_a.identity != team2_a.identity
                    and event1_a.time_slot == event2_a.time_slot
                    and event1_a.location.identity != event2_a.location.identity
                )
                or (
                    same_location
                    and team1_a.identity != team2_a.identity
                    and event1_a.location.identity == event2_a.location.identity
                    and event1_a.location.side == event2_a.location.side
                    and event1_a.time_slot != event2_a.time_slot
                )
                or (
                    not same_timeslot
                    and not same_location
                    and team1_a.identity != team2_a.identity
                    and event1_a.time_slot != event2_a.time_slot
                    and event1_a.location.identity != event2_a.location.identity
                )
            ):
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
            child,
            same_timeslot=same_timeslot,
            same_location=same_location,
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
