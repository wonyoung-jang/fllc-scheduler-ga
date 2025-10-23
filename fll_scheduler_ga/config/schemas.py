"""Pydantic models for application configuration."""

import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from ..data_model.location import Location
from ..data_model.time import TimeSlot
from .constants import CrossoverOp, MutationOp

logger = logging.getLogger(__name__)


class GaParameters(BaseModel):
    """Genetic Algorithm parameters with validation."""

    population_size: int = Field(128, gt=1)
    generations: int = Field(256, gt=0)
    offspring_size: int = Field(64, ge=0)
    crossover_chance: float = Field(0.6, ge=0.0, le=1.0)
    mutation_chance: float = Field(0.2, ge=0.0, le=1.0)
    num_islands: int = Field(1, ge=1)
    migration_interval: int = Field(0, ge=0)
    migration_size: int = Field(0, ge=0)

    def __str__(self) -> str:
        """Representation of GA parameters."""
        return (
            f"\n\tGaParameters:"
            f"\n\t  population_size    : {self.population_size}"
            f"\n\t  generations        : {self.generations}"
            f"\n\t  offspring_size     : {self.offspring_size}"
            f"\n\t  crossover_chance   : {self.crossover_chance:.2f}"
            f"\n\t  mutation_chance    : {self.mutation_chance:.2f}"
            f"\n\t  num_islands        : {self.num_islands}"
            f"\n\t  migration_interval : {self.migration_interval}"
            f"\n\t  migration_size     : {self.migration_size}"
        )


class OperatorConfig(BaseModel):
    """Configuration for genetic algorithm operators."""

    crossover_types: tuple[CrossoverOp | str, ...]
    crossover_ks: tuple[int, ...]
    mutation_types: tuple[MutationOp | str, ...]

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


class TournamentRound(BaseModel):
    """Representation of a round in the FLL tournament."""

    roundtype: str
    roundtype_idx: int
    rounds_per_team: int
    teams_per_round: int
    times: list[datetime]
    start_time: datetime
    stop_time: datetime
    duration_minutes: timedelta
    location_type: str
    locations: list[Location]
    num_timeslots: int
    timeslots: list[TimeSlot]

    def __str__(self) -> str:
        """Represent the TournamentRound."""
        return (
            f"\n\tRound:"
            f"\n\t  roundtype        : {self.roundtype}"
            f"\n\t  roundtype_idx    : {self.roundtype_idx}"
            f"\n\t  teams_per_round  : {self.teams_per_round}"
            f"\n\t  rounds_per_team  : {self.rounds_per_team}"
            f"\n\t  times            : {[str(time) for time in self.times]}"
            f"\n\t  start_time       : {self.start_time}"
            f"\n\t  stop_time        : {self.stop_time}"
            f"\n\t  duration_minutes : {self.duration_minutes}"
            f"\n\t  location         : {self.location_type}"
            f"\n\t  locations        : {[str(location) for location in self.locations]}"
            f"\n\t  num_timeslots    : {self.num_timeslots}"
            f"\n\t  timeslots        : {[str(timeslot) for timeslot in self.timeslots]}"
        )


class TournamentConfig(BaseModel):
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
    weights: tuple[float, ...]

    def __str__(self) -> str:
        """Represent the TournamentConfig."""
        return (
            f"\n  TournamentConfig:"
            f"\n    num_teams                 : {self.num_teams}"
            f"\n    time_fmt                  : {self.time_fmt}"
            f"\n    rounds                    : {[r.roundtype for r in self.rounds]}"
            f"\n    round_requirements        : {self.roundreqs}"
            f"\n    round_str_to_idx          : {self.round_str_to_idx}"
            f"\n    round_idx_to_tpr          : {self.round_idx_to_tpr}"
            f"\n    total_slots_possible      : {self.total_slots_possible}"
            f"\n    total_slots_required      : {self.total_slots_required}"
            f"\n    unique_opponents_possible : {self.unique_opponents_possible}"
            f"\n    weights                   : {self.weights}"
        )


TournamentRound.model_rebuild()
TournamentConfig.model_rebuild()
