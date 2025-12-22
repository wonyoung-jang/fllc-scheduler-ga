"""Context for the genetic algorithm parts."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.event import EventFactory, EventProperties
from ..data_model.schedule import Schedule, ScheduleContext
from ..fitness.benchmark import (
    BENCHMARKS_CACHE,
    FitnessBenchmark,
    FitnessBenchmarkBreaktime,
    FitnessBenchmarkOpponent,
    StableConfigHash,
)
from ..fitness.benchmark_repository import PickleBenchmarkRepository
from ..fitness.fitness_population import FitnessEvaluator
from ..fitness.fitness_schedule import FitnessEvaluatorSingle
from ..fitness.hard_constraint_checker import (
    HardConstraintChecker,
    HardConstraintNoRoundsNeeded,
    HardConstraintSize,
    HardConstraintTruthiness,
)
from ..io.csv_importer import CsvImporter
from ..io.ga_exporter import ScheduleSummaryGenerator
from ..io.schedule_exporter import CsvScheduleExporter
from ..io.seed_ga import GALoad, GASave, GASeedData
from ..operators.crossover import build_crossovers
from ..operators.mutation import build_mutations
from ..operators.nsga3 import NSGA3, NonDominatedSorting, ReferenceDirections
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .builder import ScheduleBuilderRandom
from .preflight_checker import PreFlightChecker

if TYPE_CHECKING:
    from ..config.app_config import AppConfig
    from ..config.schemas import ImportModel, TournamentConfig
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from ..operators.selection import Selection

logger = getLogger(__name__)


@dataclass(slots=True)
class GaContextFactory(ABC):
    """Interface for GA context factory."""

    @abstractmethod
    def build(self, app_config: AppConfig) -> GaContext:
        """Build and return a GA context."""


@dataclass(slots=True)
class StandardGaContextFactory(GaContextFactory):
    """Standard implementation of GA context factory."""

    def build(self, app_config: AppConfig) -> GaContext:
        """Build and return a GA context."""
        rng = app_config.rng
        tournament_config = app_config.tournament
        event_factory = EventFactory(tournament_config)
        event_properties = EventProperties.build(
            num_events=tournament_config.total_slots_possible,
            event_map=event_factory.as_mapping(),
        )

        # Run pre-flight checks before fitness benchmarking
        PreFlightChecker.build_then_run(event_properties, event_factory)

        roundreqs_array = np.tile(tuple(tournament_config.roundreqs.values()), (tournament_config.num_teams, 1))
        schedule_context = ScheduleContext(
            event_map=event_factory.as_mapping(),
            event_props=event_properties,
            teams_list=np.arange(tournament_config.num_teams, dtype=int),
            teams_roundreqs_arr=roundreqs_array,
            n_total_events=tournament_config.total_slots_possible,
        )
        Schedule.ctx = schedule_context

        constraints = (
            HardConstraintTruthiness(),
            HardConstraintSize(total_slots_required=tournament_config.total_slots_required),
            HardConstraintNoRoundsNeeded(),
        )
        checker = HardConstraintChecker(constraints=constraints)
        repairer = Repairer(
            config=tournament_config,
            event_factory=event_factory,
            event_properties=event_properties,
            rng=rng,
            checker=checker,
        )
        opponent_benchmarker = FitnessBenchmarkOpponent(
            config=tournament_config,
            event_factory=event_factory,
        )
        breaktime_benchmarker = FitnessBenchmarkBreaktime(
            config=tournament_config,
            event_factory=event_factory,
            model=app_config.fitness,
        )

        config_hasher = StableConfigHash(
            config=tournament_config,
            model=app_config.fitness,
        )
        config_hash = config_hasher.generate_hash()
        benchmark_cache_dir = BENCHMARKS_CACHE
        benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
        seed_file = benchmark_cache_dir / f"benchmark_cache_{config_hash}.pkl"

        repository = PickleBenchmarkRepository(path=seed_file)
        benchmark = FitnessBenchmark(
            config=tournament_config,
            model=app_config.fitness,
            repository=repository,
            opponent_benchmarker=opponent_benchmarker,
            breaktime_benchmarker=breaktime_benchmarker,
            flush_benchmarks=app_config.runtime.flush_benchmarks,
        )

        benchmark.run()
        evaluator = FitnessEvaluator(
            config=tournament_config,
            event_properties=event_properties,
            benchmark=benchmark,
            model=app_config.fitness,
        )

        ga_params = app_config.genetic.parameters
        n_objectives = evaluator.n_objectives
        ref_directions = ReferenceDirections(
            n_obj=n_objectives,
            n_pop=ga_params.population_size,
        )
        nsga3 = NSGA3(
            rng=rng,
            refs=ref_directions,
            sorting=NonDominatedSorting(),
        )

        selection = RandomSelect(rng)
        operators = app_config.genetic.operator
        crossovers = build_crossovers(rng, operators, event_factory, event_properties)
        mutations = build_mutations(rng, operators, event_factory, event_properties)

        builder = ScheduleBuilderRandom(
            event_properties=event_properties,
            rng=rng,
            round_idx_to_tpr=tournament_config.round_idx_to_tpr,
            roundtype_events=event_factory.as_roundtypes(),
        )

        ga_context_instance = GaContext(
            app_config=app_config,
            event_factory=event_factory,
            event_properties=event_properties,
            builder=builder,
            repairer=repairer,
            evaluator=evaluator,
            checker=checker,
            nsga3=nsga3,
            selection=selection,
            crossovers=crossovers,
            mutations=mutations,
        )

        RuntimeStartup(
            config=app_config,
            context=ga_context_instance,
        ).run()

        return ga_context_instance


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    event_properties: EventProperties
    evaluator: FitnessEvaluator
    checker: HardConstraintChecker
    builder: ScheduleBuilderRandom
    repairer: Repairer
    nsga3: NSGA3
    selection: Selection
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]

    def check(self, schedule: Schedule) -> bool:
        """Check a schedule using the hard constraint checker."""
        return self.checker.check(schedule)

    def repair(self, schedule: Schedule) -> bool:
        """Repair a schedule using the repairer."""
        return self.repairer.repair(schedule)

    def evaluate(self, pop_array: np.ndarray) -> tuple[np.ndarray, ...]:
        """Evaluate a schedule using the fitness evaluator."""
        return self.evaluator.evaluate(pop_array)

    def select_parents(self, n: int, k: int = 2) -> np.ndarray:
        """Select parents using the selection operator."""
        return self.selection.select(n, k)

    def select_nsga3(self, fits: np.ndarray, n_select: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Select individuals using NSGA-III."""
        return self.nsga3.select(fits, n_select)

    def get_tournament_config(self) -> TournamentConfig:
        """Get the tournament configuration from the app config."""
        return self.app_config.tournament

    def get_imports_model(self) -> ImportModel:
        """Get the imports model from the app config."""
        return self.app_config.imports

    def get_seed_island_strategy(self) -> str:
        """Get the seed island strategy from the app config."""
        return self.app_config.imports.seed_island_strategy


@dataclass(slots=True)
class RuntimeStartup:
    """Handle start of runtime with seed file handling and CSV import."""

    config: AppConfig
    context: GaContext

    def run(self) -> None:
        """Run the import schedule handler."""
        seed_file = Path(self.config.runtime.seed_file).resolve()
        if self.config.runtime.flush and seed_file.exists():
            self._flush(seed_file)
        if imported_schedule := self._import():
            self._add(seed_file, imported_schedule)

    def _flush(self, seed_file: Path) -> None:
        """Flush the seed file if specified in runtime settings."""
        seed_file.unlink(missing_ok=True)
        logger.debug("Flushed seed file at: %s", seed_file)
        seed_file.touch(exist_ok=True)

    def _import(self) -> Schedule | None:
        """Handle the import file for the genetic algorithm."""
        if not self.config.runtime.import_file:
            logger.debug("No import file specified, skipping import step.")
            return None

        import_path = Path(self.config.runtime.import_file).resolve()
        csv_importer = CsvImporter(
            import_path,
            self.config.tournament,
            self.context.event_factory,
            self.context.event_properties,
        )
        if not csv_importer.validate_inputs():
            return None

        csv_importer.run()
        imported_schedule = csv_importer.schedule
        if not self.context.checker.check(imported_schedule):
            self.context.repairer.repair(imported_schedule)

        np.array([imported_schedule.schedule], dtype=int)

        evaluator_new = FitnessEvaluatorSingle(
            config=self.config.tournament,
            event_properties=self.context.event_properties,
            benchmark=self.context.evaluator.benchmark,
            model=self.config.fitness,
        )

        if fits := evaluator_new.evaluate(imported_schedule.schedule):
            sched_fits, team_fits = fits
            imported_schedule.fitness = sched_fits
            imported_schedule.team_fitnesses = team_fits
            parent_dir = import_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            summary_gen = ScheduleSummaryGenerator(self.config.exports.team_identities)
            csv_schedule_exporter = CsvScheduleExporter(
                time_fmt=self.context.app_config.tournament.time_fmt,
                team_identities=self.config.exports.team_identities,
                event_properties=self.context.event_properties,
            )
            asyncio.run(summary_gen.export(imported_schedule, report_path))
            asyncio.run(csv_schedule_exporter.export(imported_schedule, parent_dir / "schedule.csv"))

        # if fits := self.context.evaluate(pop):
        #     sched_fits, team_fits = fits
        #     imported_schedule.fitness = sched_fits[0]
        #     imported_schedule.team_fitnesses = team_fits[0]
        #     parent_dir = import_path.parent
        #     parent_dir.mkdir(parents=True, exist_ok=True)
        #     report_path = parent_dir / "report.txt"
        #     summary_gen = ScheduleSummaryGenerator(self.config.exports.team_identities)
        #     csv_schedule_exporter = CsvScheduleExporter(
        #         time_fmt=self.context.app_config.tournament.time_fmt,
        #         team_identities=self.config.exports.team_identities,
        #         event_properties=self.context.event_properties,
        #     )
        #     asyncio.run(summary_gen.export(imported_schedule, report_path))
        #     asyncio.run(csv_schedule_exporter.export(imported_schedule, parent_dir / "schedule.csv"))

        return imported_schedule

    def _add(self, seed_file: Path, imported_schedule: Schedule) -> None:
        """Add an imported schedule to the GA population."""
        if not self.config.runtime.add_import_to_population:
            logger.debug("Not adding imported schedule to population.")
            return

        seed_data = GALoad(seed_file=seed_file, config=self.config.tournament).load()

        if seed_data is None:
            seed_data = GASeedData(config=self.config.tournament, population=[])

        if imported_schedule not in seed_data.population:
            seed_data.population.append(imported_schedule)

        GASave(seed_file=seed_file, data=seed_data).save()
