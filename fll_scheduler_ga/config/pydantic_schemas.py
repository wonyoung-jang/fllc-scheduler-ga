"""Pydantic models for application configuration."""

import logging

from pydantic import BaseModel, Field, model_validator

from fll_scheduler_ga.config.constants import CMAP_NAME_DEFAULT, OUTPUT_DIR_DEFAULT

from .constants import PICKLE_FILE_SCHEDULES, CrossoverOp, MutationOp, SeedIslandStrategy, SeedPopSort

logger = logging.getLogger(__name__)


### GeneticModel
class GaParameterModel(BaseModel):
    """Genetic Algorithm parameters."""

    population_size: int = Field(default=2, ge=2)
    generations: int = Field(default=128, ge=1)
    offspring_size: int = Field(default=1, ge=1)
    crossover_chance: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_chance: float = Field(default=0.4, ge=0.0, le=1.0)
    num_islands: int = Field(default=1, ge=1)
    migration_interval: int = Field(default=10, ge=1)
    migration_size: int = Field(default=1, ge=0)


class CrossoverModel(BaseModel):
    """Configuration for crossover operators."""

    types: tuple[CrossoverOp | str, ...] = ()
    k_vals: tuple[int, ...] = ()


class MutationModel(BaseModel):
    """Configuration for mutation operators."""

    types: tuple[MutationOp | str, ...] = ()


class OperatorModel(BaseModel):
    """Container for operator configurations."""

    crossover: CrossoverModel = Field(default_factory=CrossoverModel)
    mutation: MutationModel = Field(default_factory=MutationModel)


class StagnationModel(BaseModel):
    """Configuration for stagnation handling."""

    enable: bool = False
    proportion: float = Field(default=0.8, ge=0.0, le=1.0)
    threshold: int = 20
    cooldown: int = 50


class GeneticModel(BaseModel):
    """Configuration for the genetic algorithm."""

    rng_seed: int | str | None = None
    parameters: GaParameterModel = Field(default_factory=GaParameterModel)
    operator: OperatorModel = Field(default_factory=OperatorModel)
    stagnation: StagnationModel = Field(default_factory=StagnationModel)


class RuntimeModel(BaseModel):
    """Configuration for command-line arguments and runtime flags."""

    add_import_to_population: bool = True
    flush: bool = False
    flush_benchmarks: bool = False
    import_file: str = ""
    seed_file: str = Field(default=PICKLE_FILE_SCHEDULES, min_length=1)


### IOModel
class ImportModel(BaseModel):
    """Configuration for import options."""

    seed_pop_sort: str = SeedPopSort.RANDOM
    seed_island_strategy: str = SeedIslandStrategy.DISTRIBUTED

    @model_validator(mode="after")
    def validate(self) -> "ImportModel":
        """Validate import options."""
        if self.seed_pop_sort not in tuple(SeedPopSort):
            msg = f"Invalid seed_pop_sort: {self.seed_pop_sort}. Must be one of {[e.value for e in SeedPopSort]}."
            raise ValueError(msg)

        if self.seed_island_strategy not in tuple(SeedIslandStrategy):
            msg = (
                f"Invalid seed_island_strategy: {self.seed_island_strategy}. "
                f"Must be one of {[e.value for e in SeedIslandStrategy]}."
            )
            raise ValueError(msg)

        return self


class ExportModel(BaseModel):
    """Configuration for export options."""

    output_dir: str = Field(default=OUTPUT_DIR_DEFAULT, min_length=1)
    summary_reports: bool = True
    schedules_csv: bool = True
    schedules_html: bool = True
    schedules_team_csv: bool = True
    pareto_summary: bool = True
    plot_fitness: bool = True
    plot_parallel: bool = True
    plot_scatter: bool = True
    front_only: bool = True
    no_plotting: bool = False
    cmap_name: str = Field(default=CMAP_NAME_DEFAULT, min_length=1)
    team_identities: dict[int, str] = Field(default_factory=dict)


class IOModel(BaseModel):
    """Configuration for input/output options."""

    imports: ImportModel = Field(default_factory=ImportModel)
    exports: ExportModel = Field(default_factory=ExportModel)


### FitnessModel
class AggregationWeightsModel(BaseModel):
    """Configuration for aggregation weights."""

    mean: float | int = Field(default=1.0, ge=0.0)
    variation: float | int = Field(default=1.0, ge=0.0)
    range: float | int = Field(default=1.0, ge=0.0)
    min_fit: float = Field(default=0.3, ge=0.0)

    def get_weights_tuple(self) -> tuple[float, ...]:
        """Return the aggregation weights as a tuple."""
        weights = (self.mean, self.variation, self.range)
        total = sum(weights)
        return tuple(w / total for w in weights)


class ObjectiveWeightsModel(BaseModel):
    """Configuration for fitness objective weights."""

    breaktime: float | int = Field(default=1.0, ge=0.0)
    opponents: float | int = Field(default=1.0, ge=0.0)
    locations: float | int = Field(default=1.0, ge=0.0)

    def get_weights_tuple(self) -> tuple[float, ...]:
        """Return the objective weights as a tuple."""
        weights = (self.breaktime, self.locations, self.opponents)
        maximum = max(*weights)
        return tuple(w / maximum for w in weights)


class LocationWeightsModel(BaseModel):
    """Configuration for location weights."""

    inter_rounds: float = Field(default=0.5, ge=0.0)
    intra_rounds: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def validate(self) -> "LocationWeightsModel":
        """Validate that weights sum to 1.0."""
        total = self.inter_rounds + self.intra_rounds
        if total <= 0.0:
            self.inter_rounds = 0.5
            self.intra_rounds = 0.5
            logger.warning("Location weights sum to zero: resetting to equal weights of 0.5 each.")
        return self

    def get_weights_tuple(self) -> tuple[float, float]:
        """Return the location weights as a tuple."""
        total = self.inter_rounds + self.intra_rounds
        return (self.inter_rounds / total, self.intra_rounds / total)


class PenaltyModel(BaseModel):
    """Configuration for penalty weights."""

    zeros: float = Field(default=0.0001, lt=1.0, ge=0.0)
    minbreak: float = Field(default=0.3, lt=1.0, ge=0.0)
    minbreak_target: int = Field(default=30, gt=0)


class FitnessModel(BaseModel):
    """Configuration for fitness weights."""

    aggregation: AggregationWeightsModel = Field(default_factory=AggregationWeightsModel)
    objectives: ObjectiveWeightsModel = Field(default_factory=ObjectiveWeightsModel)
    location_weights: LocationWeightsModel = Field(default_factory=LocationWeightsModel)
    penalties: PenaltyModel = Field(default_factory=PenaltyModel)


### TournamentModel
class LocationModel(BaseModel):
    """Input model for a location type."""

    name: str = Field(default="", min_length=1)
    count: int = Field(default=1, ge=1)
    sides: int = Field(default=1, ge=1)


class RoundModel(BaseModel):
    """Input model for a tournament round."""

    roundtype: str = Field(default="", min_length=1)
    location: str = Field(default="", min_length=1)
    rounds_per_team: int = Field(default=1, ge=1)
    teams_per_round: int = Field(default=1, ge=1)
    start_time: str = ""
    stop_time: str = ""
    times: list[str] = Field(default_factory=list)
    duration_cycle: int = Field(default=0, ge=0)
    duration_active: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate(self) -> "RoundModel":
        """Validate that rounds_per_team and teams_per_round are positive."""
        if self.stop_time and not self.start_time:
            msg = f"Round '{self.roundtype}' has stop_time defined but no start_time."
            raise ValueError(msg)

        if not (self.start_time or self.times):
            msg = f"Round '{self.roundtype}' must have either start_time or times defined."
            raise ValueError(msg)

        if self.duration_active > self.duration_cycle:
            msg = f"Round '{self.roundtype}' has duration_active greater than duration_cycle."
            raise ValueError(msg)

        return self


class TournamentModel(BaseModel):
    """Root model for tournament configuration from JSON."""

    teams: int | tuple[int | str, ...] = 1
    locations: tuple[LocationModel, ...] = ()
    rounds: tuple[RoundModel, ...] = ()


### AppConfigModel
class AppConfigModel(BaseModel):
    """Root model for the entire application configuration from JSON."""

    genetic: GeneticModel = Field(default_factory=GeneticModel)
    runtime: RuntimeModel = Field(default_factory=RuntimeModel)
    io: IOModel = Field(default_factory=IOModel)
    fitness: FitnessModel = Field(default_factory=FitnessModel)
    tournament: TournamentModel = Field(default_factory=TournamentModel)
