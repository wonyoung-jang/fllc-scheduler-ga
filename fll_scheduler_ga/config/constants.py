"""Module to hold constants for the FLL Scheduler GA."""

from __future__ import annotations

from enum import StrEnum

ASCII_OFFSET = 64
FITNESS_PENALTY = 0.5
RANDOM_SEED_RANGE = (1, 2**32 - 1)
EPSILON = 1e-12


class SelectionOp(StrEnum):
    """Enum for selection operator keys."""

    RANDOM_SELECT = "RandomSelect"


class CrossoverOp(StrEnum):
    """Enum for crossover operator keys."""

    K_POINT = "KPoint"
    SCATTERED = "Scattered"
    UNIFORM = "Uniform"
    ROUND_TYPE_CROSSOVER = "RoundTypeCrossover"
    TIMESLOT_CROSSOVER = "TimeSlotCrossover"
    LOCATION_CROSSOVER = "LocationCrossover"


class MutationOp(StrEnum):
    """Enum for mutation operator keys."""

    SWAP_MATCH_CROSS_TIME_LOCATION = "SwapMatch_CrossTimeLocation"
    SWAP_MATCH_SAME_LOCATION = "SwapMatch_SameLocation"
    SWAP_MATCH_SAME_TIME = "SwapMatch_SameTime"
    SWAP_TEAM_CROSS_TIME_LOCATION = "SwapTeam_CrossTimeLocation"
    SWAP_TEAM_SAME_LOCATION = "SwapTeam_SameLocation"
    SWAP_TEAM_SAME_TIME = "SwapTeam_SameTime"
    SWAP_TABLE_SIDE = "SwapTableSide"
    INVERSION = "Inversion"
    SCRAMBLE = "Scramble"


class FitnessObjective(StrEnum):
    """Enumeration of fitness objectives for the FLL Scheduler GA."""

    BREAK_TIME = "BreakTime"
    LOCATION_CONSISTENCY = "LocationConsistency"
    OPPONENT_VARIETY = "OpponentVariety"
    # CONFLICTS = "Conflicts"


OPERATOR_CONFIG_OPTIONS = {
    ("crossover", "crossover_types", "", ""): tuple(c.value for c in CrossoverOp),
    ("crossover", "crossover_ks", "", "int"): (1, 2, 4, 8),
    ("mutation", "mutation_types", "", ""): tuple(m.value for m in MutationOp),
}
