"""Module to hold constants for the FLL Scheduler GA."""

from enum import StrEnum

ASCII_OFFSET = 64
ATTEMPTS_RANGE = (0, 50)
FITNESS_PENALTY = 0.5
HHMM_FMT = "%H:%M"
RANDOM_SEED_RANGE = (1, 2**32 - 1)


class SelectionOp(StrEnum):
    """Enum for selection operator keys."""

    TOURNAMENT_SELECT = "TournamentSelect"
    RANDOM_SELECT = "RandomSelect"


class CrossoverOp(StrEnum):
    """Enum for crossover operator keys."""

    K_POINT = "KPoint"
    SCATTERED = "Scattered"
    UNIFORM = "Uniform"
    ROUND_TYPE_CROSSOVER = "RoundTypeCrossover"
    PARTIAL_CROSSOVER = "PartialCrossover"
    BEST_TEAM_CROSSOVER = "BestTeamCrossover"


class MutationOp(StrEnum):
    """Enum for mutation operator keys."""

    SWAP_MATCH_CROSS_TIME_LOCATION = "SwapMatch_CrossTimeLocation"
    SWAP_MATCH_SAME_LOCATION = "SwapMatch_SameLocation"
    SWAP_MATCH_SAME_TIME = "SwapMatch_SameTime"
    SWAP_TEAM_CROSS_TIME_LOCATION = "SwapTeam_CrossTimeLocation"
    SWAP_TEAM_SAME_LOCATION = "SwapTeam_SameLocation"
    SWAP_TEAM_SAME_TIME = "SwapTeam_SameTime"
    SWAP_TABLE_SIDE = "SwapTableSide"


class HardConstraint(StrEnum):
    """Enumeration of hard constraints for the FLL Scheduler GA."""

    ALL_EVENTS_SCHEDULED = "AllEventsScheduled"
    SCHEDULE_EXISTENCE = "ScheduleExistence"


class FitnessObjective(StrEnum):
    """Enumeration of fitness objectives for the FLL Scheduler GA."""

    BREAK_TIME = "BreakTime"
    LOCATION_CONSISTENCY = "LocationConsistency"
    OPPONENT_VARIETY = "OpponentVariety"
