"""Module to hold constants for the FLL Scheduler GA."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

ASCII_OFFSET = 64
FITNESS_PENALTY = 0.5
RANDOM_SEED_RANGE = (1, 2**32 - 1)
EPSILON = 1e-12

CONFIG_FILE = Path("fll_scheduler_ga/config.json")

# For Importer
TIME_HEADER = "Time"
RE_HHMM = re.compile(r"\d{2}:\d{2}")

# For API
API_OUTPUT_DIR = Path("fllc_api_outputs")


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
