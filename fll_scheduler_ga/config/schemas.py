"""Pydantic models for application configuration."""

import logging

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .constants import CrossoverOp, MutationOp, SeedIslandStrategy, SeedPopSort

logger = logging.getLogger(__name__)
RANDOM_SEED_RANGE = (1, 2**32 - 1)


class GaParameters(BaseModel):
    """Genetic Algorithm parameters."""

    population_size: int = Field(default=2, ge=2)
    generations: int = Field(default=128, ge=1)
    offspring_size: int = Field(default=1, ge=1)
    crossover_chance: float = Field(default=0.7, ge=0.0, le=1.0)
    mutation_chance: float = Field(default=0.4, ge=0.0, le=1.0)
    num_islands: int = Field(default=1, ge=1)
    migration_interval: int = Field(default=10, ge=1)
    migration_size: int = Field(default=1, ge=0)
    rng_seed: int | str | None = None

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

    def get_rng_seed(self) -> int:
        """Return the RNG seed as an integer."""
        if isinstance(self.rng_seed, int):
            return self.rng_seed

        self.rng_seed = int(
            np.random.default_rng().integers(*RANDOM_SEED_RANGE)
            if self.rng_seed is None
            else abs(hash(self.rng_seed)) % (RANDOM_SEED_RANGE[1] + 1)
        )
        return self.rng_seed


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
    cooldown: int = 50


class GeneticModel(BaseModel):
    """Configuration for the genetic algorithm."""

    parameters: GaParameters
    operator: OperatorConfig
    stagnation: StagnationModel

    def get_rng_seed(self) -> int:
        """Return the RNG seed as an integer."""
        return self.parameters.get_rng_seed()


class RuntimeModel(BaseModel):
    """Configuration for command-line arguments and runtime flags."""

    add_import_to_population: bool = True
    flush: bool = False
    flush_benchmarks: bool = False
    import_file: str = ""
    seed_file: str = "fll_scheduler_ga.pkl"


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

    output_dir: str = "fllc_schedule_outputs"
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
    cmap_name: str = "viridis"
    team_identities: dict[int, str] = Field(default_factory=dict)


class TeamsModel(BaseModel):
    """Configuration for teams."""

    teams: list[int | str] | int = 1

    def __len__(self) -> int:
        """Return the number of teams."""
        if isinstance(self.teams, list):
            return len(self.teams)
        return self.teams

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
        teams_list = self.teams if isinstance(self.teams, list) else [str(i) for i in range(1, self.teams + 1)]
        return dict(enumerate(teams_list, start=1))


class FitnessModel(BaseModel):
    """Configuration for fitness weights."""

    weight_mean: float | int = 1.0
    weight_variation: float | int = 1.0
    weight_range: float | int = 1.0
    obj_weight_breaktime: float | int = 1.0
    obj_weight_opponents: float | int = 1.0
    obj_weight_locations: float | int = 1.0
    zeros_penalty: float = 0.0001
    minbreak_penalty: float = 0.1
    minbreak_target: int = 30
    min_fitness_weight: float = 0.5
    loc_weight_rounds_inter: float = 0.9
    loc_weight_rounds_intra: float = 0.1

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
            self.obj_weight_locations,
            self.obj_weight_opponents,
        )
        denom_w = max(*weights)
        return tuple(w / denom_w for w in weights)


class LocationModel(BaseModel):
    """Input model for a location type."""

    name: str = ""
    count: int = 1
    sides: int = 1


class RoundModel(BaseModel):
    """Input model for a tournament round."""

    roundtype: str = ""
    location: str = ""
    rounds_per_team: int = Field(default=1, ge=1)
    teams_per_round: int = Field(default=1, ge=1)
    start_time: str = ""
    stop_time: str = ""
    times: list[str] = Field(default_factory=list)
    duration_cycle: int = 0
    duration_active: int = 0

    @model_validator(mode="after")
    def validate(self) -> "RoundModel":
        """Validate that rounds_per_team and teams_per_round are positive."""
        if not (self.start_time or self.times):
            msg = f"Round '{self.roundtype}' must have either start_time or times defined."
            raise ValueError(msg)

        if self.stop_time and not self.start_time:
            msg = f"Round '{self.roundtype}' has stop_time defined but no start_time."
            raise ValueError(msg)

        if self.duration_active > self.duration_cycle:
            msg = f"Round '{self.roundtype}' has duration_active greater than duration_cycle."
            raise ValueError(msg)

        return self


class AppConfigModel(BaseModel):
    """Root model for the entire application configuration from JSON."""

    genetic: GeneticModel
    runtime: RuntimeModel
    imports: ImportModel
    exports: ExportModel
    teams: TeamsModel
    fitness: FitnessModel
    locations: tuple[LocationModel, ...] = ()
    rounds: tuple[RoundModel, ...] = ()
