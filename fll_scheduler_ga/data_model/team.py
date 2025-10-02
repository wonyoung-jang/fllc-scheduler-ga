"""Team data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import TournamentConfig

logger = getLogger(__name__)


@dataclass(slots=True)
class Team:
    """Data model for a team in the FLL Scheduler GA."""

    idx: int

    def __hash__(self) -> int:
        """Hash function for the team based on its identity."""
        return self.idx


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team instances."""

    config: TournamentConfig
    team_array: np.ndarray[Team] = None

    def __post_init__(self) -> None:
        """Post-initialization to set up the initial state."""
        self.team_array = np.array([Team(idx=i) for i in range(self.config.num_teams)])

    def build(self) -> np.ndarray[Team]:
        """Create a list of Team instances.

        Returns:
            np.ndarray[Team]: An array of Team instances.

        """
        return self.team_array
