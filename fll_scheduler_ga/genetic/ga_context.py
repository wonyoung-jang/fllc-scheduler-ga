"""Context for the genetic algorithm parts."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.event import EventFactory, EventProperties
from ..data_model.schedule import Schedule, ScheduleContext
from ..fitness.benchmark import FitnessBenchmark
from ..fitness.fitness import FitnessEvaluator, HardConstraintChecker
from ..io.csv_importer import CsvImporter
from ..io.ga_exporter import ScheduleSummaryGenerator
from ..io.seed_ga import GALoad, GASave, GASeedData
from ..operators.crossover import build_crossovers
from ..operators.mutation import build_mutations
from ..operators.nsga3 import NSGA3, ReferenceDirections
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .builder import ScheduleBuilder
from .preflight_checker import PreFlightChecker

if TYPE_CHECKING:
    from ..config.app_config import AppConfig
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from ..operators.selection import Selection

logger = getLogger(__name__)


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    event_properties: EventProperties
    evaluator: FitnessEvaluator
    checker: HardConstraintChecker
    builder: ScheduleBuilder
    repairer: Repairer
    nsga3: NSGA3
    selection: Selection
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]

    @classmethod
    def build(cls, app_config: AppConfig) -> GaContext:
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

        checker = HardConstraintChecker(tournament_config)
        repairer = Repairer(
            config=tournament_config,
            event_factory=event_factory,
            event_properties=event_properties,
            rng=rng,
            checker=checker,
        )
        benchmark = FitnessBenchmark(
            config=tournament_config,
            event_factory=event_factory,
            event_properties=event_properties,
            model=app_config.fitness,
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
        )

        selection = RandomSelect(rng)
        operators = app_config.genetic.operator
        crossovers = build_crossovers(rng, operators, event_factory, event_properties)
        mutations = build_mutations(rng, operators, event_factory, event_properties)

        builder = ScheduleBuilder(
            event_factory=event_factory,
            event_properties=event_properties,
            config=tournament_config,
            rng=rng,
        )

        ga_context_instance = cls(
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
class RuntimeStartup:
    """Handle start of runtime with seed file handling and CSV import."""

    config: AppConfig
    context: GaContext

    def run(self) -> None:
        """Run the import schedule handler."""
        seed_file = Path(self.config.runtime.seed_file).resolve()
        self._flush(seed_file)
        if imported_schedule := self._import():
            self._add(seed_file, imported_schedule)

    def _flush(self, seed_file: Path) -> None:
        """Flush the seed file if specified in runtime settings."""
        if self.config.runtime.flush and seed_file.exists():
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
        pop = np.array([imported_schedule.schedule], dtype=int)

        if fits := self.context.evaluator.evaluate_population(pop):
            sched_fits, team_fits = fits
            imported_schedule.fitness = sched_fits[0]
            imported_schedule.team_fitnesses = team_fits[0]
            parent_dir = import_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            summary_gen = ScheduleSummaryGenerator(self.config.exports.team_identities)
            summary_gen.export(imported_schedule, report_path)

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
