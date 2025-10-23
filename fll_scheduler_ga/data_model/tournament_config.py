"""Configuration for the tournament."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tournament_round import TournamentRound

logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    time_fmt: str
    rounds: list[TournamentRound]
    roundreqs: dict[str, int]
    round_str_to_idx: dict[str, int]
    round_idx_to_tpr: dict[int, int]
    total_slots_possible: int
    total_slots_required: int
    unique_opponents_possible: bool
    weights: tuple[float, float]

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        logger.debug("Tournament configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.roundtype}" for r in self.rounds)
        roundreqs_str = ", ".join(f"{k}: {v}" for k, v in self.roundreqs.items())

        return (
            f"\n  TournamentConfig:"
            f"\n    num_teams                 : {self.num_teams}"
            f"\n    time_fmt                  : {self.time_fmt}"
            f"\n    rounds                    : {rounds_str}"
            f"\n    round_requirements        : {roundreqs_str}"
            f"\n    round_str_to_idx          : {self.round_str_to_idx}"
            f"\n    round_idx_to_tpr          : {self.round_idx_to_tpr}"
            f"\n    total_slots_possible      : {self.total_slots_possible}"
            f"\n    total_slots_required      : {self.total_slots_required}"
            f"\n    unique_opponents_possible : {self.unique_opponents_possible}"
            f"\n    weights                   : {self.weights}"
        )
