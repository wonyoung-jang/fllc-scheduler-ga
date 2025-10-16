"""Configuration for genetic algorithm operators."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config.constants import CrossoverOp, MutationOp

logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OperatorConfig:
    """Configuration for the genetic algorithm operators."""

    crossover_types: tuple[CrossoverOp | str]
    crossover_ks: tuple[int]
    mutation_types: tuple[MutationOp | str]

    def __post_init__(self) -> None:
        """Post-initialization for operator configuration."""
        logger.debug("Operator configuration loaded: %s", self)

    def __str__(self) -> str:
        """Represent the OperatorConfig."""
        crossovers_str = f"{'\n\t\t'.join(str(c) for c in self.crossover_types)}"
        crossover_ks_str = f"{'\n\t\t'.join(str(k) for k in self.crossover_ks)}"
        mutations_str = f"{'\n\t\t'.join(str(m) for m in self.mutation_types)}"
        return (
            f"\n\tOperatorConfig:"
            f"\n\t  crossover_types:\n\t\t{crossovers_str}"
            f"\n\t  crossover_ks:\n\t\t{crossover_ks_str}"
            f"\n\t  mutation_types:\n\t\t{mutations_str}"
        )

    @classmethod
    def build(cls, params: dict[str, Any]) -> OperatorConfig:
        """Build an OperatorConfig from a dictionary of parameters."""
        return cls(**params)
