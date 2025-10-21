"""Configuration for the tournament."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .location import Location
    from .tournament_round import TournamentRound

logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    time_fmt: str
    rounds: list[TournamentRound]
    round_requirements: dict[str, int]
    round_to_int: dict[str, int]
    round_to_tpr: dict[str, int]
    round_idx_to_tpr: dict[int, int]
    total_slots_possible: int
    total_slots_required: int
    unique_opponents_possible: bool
    weights: tuple[float, float]
    locations: list[Location]

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        logger.debug("Tournament configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.roundtype}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())

        return (
            f"\n\tTournamentConfig:"
            f"\n\t  num_teams                 : {self.num_teams}"
            f"\n\t  time_fmt                  : {self.time_fmt}"
            f"\n\t  rounds                    : {rounds_str}"
            f"\n\t  round_requirements        : {round_reqs_str}"
            f"\n\t  round_to_int              : {self.round_to_int}"
            f"\n\t  round_to_tpr              : {self.round_to_tpr}"
            f"\n\t  round_idx_to_tpr          : {self.round_idx_to_tpr}"
            f"\n\t  total_slots_possible      : {self.total_slots_possible}"
            f"\n\t  total_slots_required      : {self.total_slots_required}"
            f"\n\t  unique_opponents_possible : {self.unique_opponents_possible}"
            f"\n\t  weights                   : {self.weights}"
            f"\n\t  locations                 : {[str(loc) for loc in self.locations]}"
        )
