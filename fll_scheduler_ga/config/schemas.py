"""Pydantic models for application configuration."""

import logging
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..data_model.location import Location
from ..data_model.timeslot import TimeSlot
from .constants import RANDOM_SEED_RANGE, CrossoverOp, MutationOp, SeedIslandStrategy, SeedPopSort

logger = logging.getLogger(__name__)


class GaParameters(BaseModel):
    """Genetic Algorithm parameters."""

    population_size: int
    generations: int
    offspring_size: int
    crossover_chance: float
    mutation_chance: float
    num_islands: int
    migration_interval: int
    migration_size: int
    rng_seed: int | str | None

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
            f"\n\t  migrate_interval   : {self.migration_interval}"
            f"\n\t  migrate_size       : {self.migration_size}"
            f"\n\t  rng_seed           : {self.rng_seed}"
        )

    @model_validator(mode="after")
    def validate(self) -> "GaParameters":
        """Validate that migration settings are only used with multiple islands."""
        if self.generations < 1:
            self.generations = 128
            logger.warning("Generations must be at least 1, defaulting to %d.", self.generations)

        if self.population_size < 2:
            self.population_size = 2
            logger.warning("Population size must be at least 2, defaulting to %d.", self.population_size)

        if self.offspring_size < 1:
            self.offspring_size = 1
            logger.warning("Offspring size must be at least 1, defaulting to %d.", self.offspring_size)

        if not (0.0 < self.crossover_chance <= 1.0):
            self.crossover_chance = 0.7
            logger.warning("Crossover chance must be between 0.0 and 1.0, defaulting to %f.", self.crossover_chance)

        if not (0.0 <= self.mutation_chance <= 1.0):
            self.mutation_chance = 0.4
            logger.warning("Mutation chance must be between 0.0 and 1.0, defaulting to %f.", self.mutation_chance)

        if self.num_islands < 1:
            self.num_islands = 1
            logger.warning("Number of islands must be at least 1, defaulting to %d.", self.num_islands)

        if self.migration_interval < 1:
            self.migration_interval = self.generations // 10 or 1
            logger.warning("Migration interval must be at least 1, defaulting to %d.", self.migration_interval)

        if self.migration_size < 0:
            self.migration_size = 0
            logger.warning("Migration size cannot be negative, defaulting to %d.", self.migration_size)

        if self.num_islands > 1 and self.migration_size >= self.population_size:
            self.migration_size = max(1, self.population_size // 5)
            logger.warning("Migration size is >= population size, defaulting to max(1, 20%%): %i", self.migration_size)

        if self.rng_seed is None:
            sv = np.random.default_rng().integers(*RANDOM_SEED_RANGE)
            if not isinstance(sv, int):
                self.rng_seed = abs(hash(sv)) % (RANDOM_SEED_RANGE[1] + 1)
            else:
                self.rng_seed = sv
            logger.info("RNG seed not set, defaulting to %s.", self.rng_seed)

        return self


class CrossoverModel(BaseModel):
    """Configuration for crossover operators."""

    types: list[CrossoverOp | str] = Field(default_factory=list)
    k_vals: list[int] = Field(default_factory=list)


class MutationModel(BaseModel):
    """Configuration for mutation operators."""

    types: list[MutationOp | str] = Field(default_factory=list)


class OperatorConfig(BaseModel):
    """Container for operator configurations."""

    crossover: CrossoverModel
    mutation: MutationModel

    def __str__(self) -> str:
        """Represent the OperatorConfig."""
        return (
            f"\n\tOperatorConfig:"
            f"\n\t  crossover_types:\n\t\t{'\n\t\t'.join(str(c) for c in self.crossover.types)}"
            f"\n\t  crossover_ks:\n\t\t{'\n\t\t'.join(str(k) for k in self.crossover.k_vals)}"
            f"\n\t  mutation_types:\n\t\t{'\n\t\t'.join(str(m) for m in self.mutation.types)}"
        )


class StagnationModel(BaseModel):
    """Configuration for stagnation handling."""

    enable: bool = False
    proportion: float = 0.8
    threshold: int = 20


class GeneticModel(BaseModel):
    """Configuration for the genetic algorithm."""

    parameters: GaParameters
    operator: OperatorConfig
    stagnation: StagnationModel


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


class ImportModel(BaseModel):
    """Configuration for import options."""

    seed_pop_sort: str = "random"
    seed_island_strategy: str = "distributed"

    @model_validator(mode="after")
    def validate(self) -> "ImportModel":
        """Validate import options."""
        if self.seed_pop_sort not in list(SeedPopSort):
            msg = f"Invalid seed_pop_sort: {self.seed_pop_sort}. Must be one of {[e.value for e in SeedPopSort]}."
            raise ValueError(msg)

        if self.seed_island_strategy not in list(SeedIslandStrategy):
            msg = (
                f"Invalid seed_island_strategy: {self.seed_island_strategy}. "
                f"Must be one of {[e.value for e in SeedIslandStrategy]}."
            )
            raise ValueError(msg)

        return self


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
    team_identities: dict[int, str] = Field(default_factory=dict)


class TeamsModel(BaseModel):
    """Configuration for teams."""

    teams: list[int | str] | int

    def __len__(self) -> int:
        """Return the number of teams."""
        return len(self.teams)

    @model_validator(mode="after")
    def validate(self) -> "TeamsModel":
        """Validate that num_teams matches the length of identities if both are provided."""
        if isinstance(self.teams, list):
            self.teams = [str(t) for t in self.teams]
        elif isinstance(self.teams, int):
            self.teams = [str(i) for i in range(1, self.teams + 1)]
        return self

    def get_team_ids(self) -> dict[int, str]:
        """Return a mapping of team indices to team identities."""
        return dict(enumerate(self.teams, start=1))


class FitnessModel(BaseModel):
    """Configuration for fitness weights."""

    weight_mean: int
    weight_variation: int
    weight_range: int
    obj_weight_breaktime: int = 1
    obj_weight_opponents: int = 1
    obj_weight_locations: int = 1
    zeros_penalty: float = 0.0001
    minbreak_penalty: float = 0.1
    minbreak_target: int = 30
    min_fitness_weight: float = 0.5

    @model_validator(mode="after")
    def validate(self) -> "FitnessModel":
        """Validate that fitness weights are non-negative."""
        self.weight_mean = max(0.0, self.weight_mean)
        self.weight_variation = max(0.0, self.weight_variation)
        self.weight_range = max(0.0, self.weight_range)
        weights = (
            self.weight_mean,
            self.weight_variation,
            self.weight_range,
        )
        if all(w == 0.0 for w in weights):
            self.weight_mean = 1.0
            self.weight_variation = 1.0
            self.weight_range = 1.0
            logger.warning("All fitness weights were zero; defaulting all weights to 1.0.")

        if self.minbreak_penalty <= 0.0:
            self.minbreak_penalty = 0.1
            logger.warning("minbreak_penalty must be positive; defaulting to 0.1.")

        return self

    def get_fitness_tuple(self) -> tuple[float, ...]:
        """Return the fitness weights as a tuple."""
        weights = (
            self.weight_mean,
            self.weight_variation,
            self.weight_range,
        )
        sum_w = sum(weights)
        return tuple(w / sum_w for w in weights)

    def get_obj_weights(self) -> tuple[float, ...]:
        """Return the objective weights as a tuple."""
        weights = (
            self.obj_weight_breaktime,
            self.obj_weight_opponents,
            self.obj_weight_locations,
        )
        denom_w = max(*weights)
        return tuple(w / denom_w for w in weights)


class LocationModel(BaseModel):
    """Input model for a location type."""

    name: str
    count: int
    sides: int


class RoundModel(BaseModel):
    """Input model for a tournament round."""

    roundtype: str
    location: str
    rounds_per_team: int = Field(default=1, ge=1)
    teams_per_round: int = Field(default=1, ge=1)
    start_time: str = ""
    stop_time: str = ""
    times: list[str] = Field(default_factory=list)
    duration_minutes: int = 0

    @model_validator(mode="after")
    def validate(self) -> "RoundModel":
        """Validate that rounds_per_team and teams_per_round are positive."""
        if not (self.start_time or self.times):
            msg = f"Round '{self.roundtype}' must have either start_time or times defined."
            raise ValueError(msg)

        if self.stop_time and not self.start_time:
            msg = f"Round '{self.roundtype}' has stop_time defined but no start_time."
            raise ValueError(msg)

        return self


class AppConfigModel(BaseModel):
    """Root model for the entire application configuration from JSON."""

    genetic: GeneticModel
    runtime: RuntimeModel
    imports: ImportModel
    exports: ExportModel
    logging: LoggingModel
    teams: TeamsModel
    fitness: FitnessModel
    locations: list[LocationModel]
    rounds: list[RoundModel]


class TournamentRound(BaseModel):
    """Representation of a round in the FLL tournament."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, frozen=True)
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
    slots_empty: int
    unfilled_allowed: bool

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
            f"\n\t  slots_empty      : {self.slots_empty}"
            f"\n\t  unfilled_allowed : {self.unfilled_allowed}"
        )

    @field_validator("slots_empty", mode="after")
    @classmethod
    def validate_slots_empty(cls, v: int) -> int:
        """Validate that slots_empty is not negative."""
        if v < 0:
            msg = (
                "Insufficient capacity for TournamentRound (required > available).\n"
                "Suggestion: increase number of locations or timeslots."
            )
            raise ValueError(msg)
        return v


class TournamentConfig(BaseModel):
    """Configuration for the tournament."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, frozen=True)
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
    is_interleaved: bool

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
            f"\n    is_interleaved            : {self.is_interleaved}"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality between two TournamentConfig instances."""
        if not isinstance(other, TournamentConfig):
            return NotImplemented

        return (
            self.num_teams == other.num_teams
            and self.time_fmt == other.time_fmt
            and self.rounds == other.rounds
            and self.roundreqs == other.roundreqs
            and self.round_str_to_idx == other.round_str_to_idx
            and self.round_idx_to_tpr == other.round_idx_to_tpr
            and self.total_slots_possible == other.total_slots_possible
            and self.total_slots_required == other.total_slots_required
            and self.unique_opponents_possible == other.unique_opponents_possible
            and self.all_locations == other.all_locations
            and self.all_timeslots == other.all_timeslots
            and self.max_events_per_team == other.max_events_per_team
            and self.is_interleaved == other.is_interleaved
        )

    def __hash__(self) -> int:
        """Generate a hash for the TournamentConfig."""
        return super().__hash__()


TournamentRound.model_rebuild()
TournamentConfig.model_rebuild()
