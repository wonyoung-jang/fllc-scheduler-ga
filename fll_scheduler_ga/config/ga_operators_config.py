"""Configuration for genetic algorithm operators."""

from dataclasses import dataclass

from ..config.constants import CrossoverOps, MutationOps, SelectionOps


@dataclass(slots=True, frozen=True)
class OperatorConfig:
    """Configuration for the genetic algorithm operators."""

    selection_types: list[SelectionOps | str]
    crossover_types: list[CrossoverOps | str]
    crossover_ks: list[int]
    mutation_types: list[MutationOps | str]

    def __str__(self) -> str:
        """Represent the OperatorConfig."""
        selections_str = f"{'\n    - '.join(str(s) for s in self.selection_types)}"
        crossovers_str = f"{'\n    - '.join(str(c) for c in self.crossover_types)}"
        crossover_ks_str = f"{'\n    - '.join(str(k) for k in self.crossover_ks)}"
        mutations_str = f"{'\n    - '.join(str(m) for m in self.mutation_types)}"

        return (
            "OperatorConfig:\n"
            f"  Selection Types:\n    - {selections_str}\n"
            f"  Crossover Types:\n    - {crossovers_str}\n"
            f"  Crossover K-values:\n    - {crossover_ks_str}\n"
            f"  Mutation Types:\n    - {mutations_str}"
        )
