"""Pydantic models for application configuration."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator

from ..data_model.location import Location
from ..data_model.time import TimeSlot
from .constants import CrossoverOp, MutationOp

logger = logging.getLogger(__name__)


class GaParametersModel(BaseModel):
    """Genetic Algorithm parameters."""

    population_size: int = Field(128, gt=1)
    generations: int = Field(256, gt=0)
    offspring_size: int = Field(64, ge=0)
    crossover_chance: float = Field(0.6, ge=0.0, le=1.0)
    mutation_chance: float = Field(0.2, ge=0.0, le=1.0)
    num_islands: int = Field(1, ge=1)
    migration_interval: int = Field(0, ge=0)
    migration_size: int = Field(0, ge=0)
    rng_seed: int | str | None


class CrossoverModel(BaseModel):
    """Configuration for crossover operators."""

    types: list[CrossoverOp | str] = Field(default_factory=list)
    k_vals: list[int] = Field(default_factory=list)


class MutationModel(BaseModel):
    """Configuration for mutation operators."""

    types: list[MutationOp | str] = Field(default_factory=list)


class OperatorModel(BaseModel):
    """Container for operator configurations."""

    crossover: CrossoverModel
    mutation: MutationModel


class GeneticModel(BaseModel):
    """Configuration for the genetic algorithm."""

    parameters: GaParametersModel
    operator: OperatorModel


class RuntimeModel(BaseModel):
    """Configuration for command-line arguments and runtime flags."""

    add_import_to_population: bool
    flush: bool
    flush_benchmarks: bool
    import_file: str
    seed_file: str


class LoggingModel(BaseModel):
    """Configuration for logging."""

    log_file: str
    loglevel_file: str
    loglevel_console: str


class ExportModel(BaseModel):
    """Configuration for export options."""

    output_dir: str
    summary_reports: bool
    schedules_csv: bool
    schedules_html: bool
    schedules_team_csv: bool
    pareto_summary: bool
    plot_fitness: bool
    plot_parallel: bool
    plot_scatter: bool
    front_only: bool
    no_plotting: bool
    cmap_name: str


class TeamsModel(BaseModel):
    """Configuration for teams."""

    num_teams: int | None
    identities: list[int | str]


class FitnessModel(BaseModel):
    """Configuration for fitness weights."""

    weight_mean: float = 1.0
    weight_variation: float = 1.0
    weight_range: float = 1.0


class TimeModel(BaseModel):
    """Configuration for time format."""

    format: int

    @field_validator("format")
    @classmethod
    def check_format(cls, v: int) -> int:
        """Validate that the time format is either 12 or 24."""
        if v not in (12, 24):
            msg = "Invalid time format. Must be 12 or 24."
            raise ValueError(msg)
        return v


class LocationModel(BaseModel):
    """Input model for a location type."""

    name: str
    count: int
    sides: int


class RoundModel(BaseModel):
    """Input model for a tournament round."""

    roundtype: str
    location: str
    rounds_per_team: int
    teams_per_round: int
    start_time: str
    stop_time: str | None
    times: list[str]
    duration_minutes: int


class AppConfigModel(BaseModel):
    """Root model for the entire application configuration from JSON."""

    genetic: GeneticModel
    runtime: RuntimeModel
    exports: ExportModel
    logging: LoggingModel
    teams: TeamsModel
    fitness: FitnessModel
    time: TimeModel
    locations: list[LocationModel]
    rounds: list[RoundModel]


@dataclass(slots=True)
class GaParameters:
    """Genetic Algorithm parameters."""

    population_size: int
    generations: int
    offspring_size: int
    crossover_chance: float
    mutation_chance: float
    num_islands: int
    migrate_interval: int
    migrate_size: int

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
            f"\n\t  migrate_interval   : {self.migrate_interval}"
            f"\n\t  migrate_size       : {self.migrate_size}"
        )


@dataclass(slots=True)
class OperatorConfig:
    """Genetic Algorithm operators."""

    crossover_types: list[CrossoverOp | str]
    crossover_ks: list[int]
    mutation_types: list[MutationOp | str]

    def __str__(self) -> str:
        """Represent the OperatorConfig."""
        return (
            f"\n\tOperatorConfig:"
            f"\n\t  crossover_types:\n\t\t{'\n\t\t'.join(str(c) for c in self.crossover_types)}"
            f"\n\t  crossover_ks:\n\t\t{'\n\t\t'.join(str(k) for k in self.crossover_ks)}"
            f"\n\t  mutation_types:\n\t\t{'\n\t\t'.join(str(m) for m in self.mutation_types)}"
        )


@dataclass(slots=True)
class TournamentRound:
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
    slots_total: int
    slots_required: int

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
            f"\n\t  slots_total      : {self.slots_total}"
            f"\n\t  slots_required   : {self.slots_required}"
        )


@dataclass(slots=True)
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
    weights: tuple[float, ...]
    all_locations: list[Location]
    all_timeslots: list[TimeSlot]
    max_events_per_team: int

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
            f"\n    all_locations             : {[str(loc) for loc in self.all_locations]}"
            f"\n    all_timeslots             : {[str(ts) for ts in self.all_timeslots]}"
            f"\n    max_events_per_team       : {self.max_events_per_team}"
        )
