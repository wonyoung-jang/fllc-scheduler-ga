"""Configuration for the tournament scheduler GA."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config.constants import CrossoverOps, MutationOps, SelectionOps

logger = logging.getLogger(__name__)

type RoundType = str


@dataclass(slots=True, frozen=True)
class Round:
    """Representation of a round in the FLL tournament."""

    round_type: RoundType
    rounds_per_team: int
    teams_per_round: int
    times: list[datetime]
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    num_locations: int
    num_teams: int

    def get_num_slots(self) -> int:
        """Get the number of slots available for this round."""
        if self.times:
            return len(self.times)

        total_num_teams = self.num_teams * self.rounds_per_team
        slots_per_timeslot = self.num_locations * self.teams_per_round

        if slots_per_timeslot == 0:
            return 0

        minimum_slots = math.ceil(total_num_teams / slots_per_timeslot)

        if self.stop_time:
            total_available = self.stop_time - self.start_time
            slots_in_window = int(total_available / self.duration_minutes)
            return max(minimum_slots, slots_in_window)

        return minimum_slots


@dataclass(slots=True, frozen=True)
class TournamentConfig:
    """Configuration for the tournament."""

    num_teams: int
    rounds: list[Round]
    round_requirements: dict[RoundType, int]
    total_slots: int
    unique_opponents_possible: bool

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        rounds_str = ", ".join(f"{r.round_type}" for r in sorted(self.rounds, key=lambda x: x.start_time))
        round_reqs_str = ", ".join(f"{k}: {v}" for k, v in self.round_requirements.items())

        return (
            f"TournamentConfig:\n"
            f"\tNumber of Teams: {self.num_teams}\n"
            f"\tRound Types: {rounds_str}\n"
            f"\tRound Requirements: {round_reqs_str}\n"
            f"\tTotal Slots: {self.total_slots}\n"
            f"\tUnique Opponents Possible: {self.unique_opponents_possible}"
        )


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
