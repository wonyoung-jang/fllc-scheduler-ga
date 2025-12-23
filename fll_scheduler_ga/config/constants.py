"""Module to hold constants for the scheduler."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

ASCII_OFFSET = 64
EPSILON = 1e-12
MAIN_PACKAGE_DIR = Path("fll_scheduler_ga").resolve()
CONFIG_FILE_DEFAULT = MAIN_PACKAGE_DIR / "config.json"
LOGGING_CONFIG_PATH = MAIN_PACKAGE_DIR / "logging.json"


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
    """Enumeration of fitness objectives for the scheduler."""

    BREAK_TIME = "BreakTime"
    LOCATION_CONSISTENCY = "LocationConsistency"
    OPPONENT_VARIETY = "OpponentVariety"


class SeedPopSort(StrEnum):
    """Enumeration of seed population sorting strategies."""

    RANDOM = "random"
    BEST = "best"


class SeedIslandStrategy(StrEnum):
    """Enumeration of seed island distribution strategies."""

    DISTRIBUTED = "distributed"
    CONCENTRATED = "concentrated"
