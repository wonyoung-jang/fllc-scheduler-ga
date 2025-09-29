"""Team data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import RoundType, TournamentConfig
    from .event import Event

logger = getLogger(__name__)


@dataclass(slots=True)
class Team:
    """Data model for a team in the FLL Scheduler GA."""

    idx: int
    roundreqs: dict[RoundType, int]
    fitness: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=float), repr=False)
    events: set[int] = field(default_factory=set, repr=False)

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return self.idx

    def clone(self) -> Team:
        """Create a deep copy of the team."""
        return Team(
            idx=self.idx,
            roundreqs=self.roundreqs.copy(),
            fitness=self.fitness.copy(),
            events=self.events.copy(),
        )

    def rounds_needed(self) -> bool:
        """Check if any rounds still needed for the team."""
        return sum(self.roundreqs.values()) > 0

    def needs_round(self, round_type: RoundType) -> bool:
        """Check if the team still needs to participate in a given round type."""
        return self.roundreqs[round_type] > 0

    def switch_event(self, old_event: Event, new_event: Event) -> None:
        """Switch an event for the team."""
        self.remove_event(old_event)
        self.add_event(new_event)

    def remove_event(self, event: Event) -> None:
        """Unbook a team from an event."""
        self.roundreqs[event.roundtype] += 1
        self.events.remove(event.idx)

    def add_event(self, event: Event) -> None:
        """Book a team for an event."""
        self.roundreqs[event.roundtype] -= 1
        self.events.add(event.idx)

    def conflicts(self, new_event: Event, *, ignore: Event = None) -> bool:
        """Check if adding a new event would cause a time conflict.

        Args:
            new_event (Event): The new event to check for conflicts.
            ignore (Event): An event to ignore when checking for conflicts.

        Returns:
            bool: True if there is a conflict, False otherwise.

        """
        if ignore and ignore.idx in self.events:
            self.events.remove(ignore.idx)

        conflict_found = new_event.idx in self.events

        if not conflict_found and new_event.conflicts and not self.events.isdisjoint(new_event.conflicts):
            conflict_found = True

        if ignore:
            self.events.add(ignore.idx)

        return conflict_found


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig
    team_ids_raw: np.ndarray[int] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.team_ids_raw = np.arange(self.config.num_teams)

    def build(self) -> np.ndarray[Team]:
        """Create a list of Team instances.

        Returns:
            np.ndarray[Team]: An array of Team instances.

        """
        return np.asarray(
            [
                Team(
                    idx=i,
                    roundreqs=self.config.round_requirements.copy(),
                )
                for i in self.team_ids_raw
            ]
        )
