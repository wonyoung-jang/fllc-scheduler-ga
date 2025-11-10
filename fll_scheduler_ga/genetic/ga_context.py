"""Context for the genetic algorithm parts."""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..data_model.event import EventFactory, EventProperties
from ..data_model.schedule import Schedule
from ..fitness.benchmark import FitnessBenchmark
from ..fitness.fitness import FitnessEvaluator, HardConstraintChecker
from ..io.csv_importer import CsvImporter
from ..io.ga_exporter import ScheduleSummaryGenerator
from ..operators.crossover import build_crossovers
from ..operators.mutation import build_mutations
from ..operators.nsga3 import NSGA3, ReferenceDirections
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .builder import ScheduleBuilder

if TYPE_CHECKING:
    from ..config.app_config import AppConfig
    from ..data_model.timeslot import TimeSlot
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

    def __post_init__(self) -> None:
        """Post-initialization actions."""
        self.handle_import_file()

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
        PreFlightChecker(
            event_factory=event_factory,
            event_properties=event_properties,
        ).run_checks()

        Schedule.teams = np.arange(tournament_config.num_teams, dtype=int)
        Schedule.event_properties = event_properties
        Schedule.event_map = event_factory.as_mapping()

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
            flush_benchmarks=app_config.runtime.flush_benchmarks,
        )
        evaluator = FitnessEvaluator(
            config=tournament_config,
            event_properties=event_properties,
            benchmark=benchmark,
        )

        ga_params = app_config.ga_params
        n_total_pop = ga_params.population_size * ga_params.num_islands
        n_objectives = len(evaluator.objectives)
        ref_directions = ReferenceDirections(
            n_obj=n_objectives,
            n_pop=n_total_pop,
        )
        nsga3 = NSGA3(
            rng=rng,
            refs=ref_directions,
        )

        selection = RandomSelect(rng)
        operators = app_config.operators
        crossovers = build_crossovers(rng, operators, event_factory, event_properties)
        mutations = build_mutations(rng, operators, event_factory, event_properties)

        builder = ScheduleBuilder(
            event_factory=event_factory,
            event_properties=event_properties,
            config=tournament_config,
            rng=rng,
        )

        return cls(
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

    def handle_import_file(self) -> None:
        """Handle the import file for the genetic algorithm."""
        runtime = self.app_config.runtime
        config = self.app_config.tournament
        path = Path(runtime.seed_file).resolve()

        if runtime.flush and path.exists():
            path.unlink(missing_ok=True)
        path.touch(exist_ok=True)

        if not runtime.import_file:
            logger.debug("No import file specified, skipping import step.")
            return

        import_path = Path(runtime.import_file).resolve()
        csv_importer = CsvImporter(import_path, config, self.event_factory, self.event_properties)
        imported_schedule_csv = csv_importer.schedule

        pop = np.array([imported_schedule_csv.schedule], dtype=int)

        if fits := self.evaluator.evaluate_population(pop):
            sched_fits, team_fits = fits
            imported_schedule_csv.fitness = sched_fits[0]
            imported_schedule_csv.team_fitnesses = team_fits[0]
            parent_dir = import_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            summary_gen = ScheduleSummaryGenerator()
            summary_gen.export(imported_schedule_csv, report_path)

        if not runtime.add_import_to_population:
            logger.debug("Not adding imported schedule to population.")
            return

        population = []
        try:
            with path.open("rb") as f:
                seed_data = pickle.load(f)
                population.extend(seed_data.get("population", []))
        except (OSError, pickle.PicklingError):
            logger.exception("Error loading seed file")
        except EOFError:
            logger.debug("Pickle file is empty")

        if imported_schedule_csv not in population:
            population.append(imported_schedule_csv)

        try:
            with path.open("wb") as f:
                data_to_cache = {
                    "population": population,
                    "config": self.app_config.tournament,
                }
                pickle.dump(data_to_cache, f)
        except (OSError, pickle.PicklingError):
            logger.exception("Error loading seed file")
        except EOFError:
            logger.debug("Pickle file is empty")


@dataclass(slots=True)
class PreFlightChecker:
    """Run pre-flight checks on the tournament configuration."""

    event_properties: EventProperties
    event_factory: EventFactory

    def run_checks(self) -> None:
        """Run all pre-flight checks."""
        try:
            self.check_location_time_overlaps()
        except ValueError:
            logger.exception("Preflight checks failed. Please review the configuration.")
        logger.debug("All preflight checks passed successfully.")

    def check_location_time_overlaps(self) -> None:
        """Check if different round types are scheduled in the same locations at the same time."""
        ep = self.event_properties
        booked_slots: dict[int, list[tuple[TimeSlot, str]]] = defaultdict(list)
        for e in self.event_factory.build_indices():
            loc_str = ep.loc_str[e]
            loc_idx = ep.loc_idx[e]
            ts = ep.timeslot[e]
            rt = ep.roundtype[e]
            for existing_ts, existing_rt in booked_slots.get(loc_idx, []):
                if ts.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: TournamentRound '{rt}' and '{existing_rt}' "
                        f"are scheduled in the same location ({loc_str} {loc_idx}) "
                        f"at overlapping times ({ts} and "
                        f"{existing_ts})."
                    )
                    raise ValueError(msg)
            booked_slots[loc_idx].append((ts, rt))
        logger.debug("Check passed: No location/time overlaps found.")
