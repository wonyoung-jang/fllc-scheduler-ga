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
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .builder import ScheduleBuilder

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

    def __post_init__(self) -> None:
        """Post-initialization to validate the GA context."""
        self.run_preflight_checks()

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

        Schedule.teams = np.arange(tournament_config.num_teams, dtype=int)
        Schedule.conflict_matrix = event_factory.build_conflict_matrix()
        Schedule.event_properties = event_properties
        Schedule.event_map = event_factory.as_mapping()

        repairer = Repairer(
            rng,
            tournament_config,
            event_factory,
            event_properties,
        )
        benchmark = FitnessBenchmark(
            tournament_config,
            event_factory,
            event_properties,
            flush_benchmarks=app_config.arguments.flush_benchmarks,
        )
        evaluator = FitnessEvaluator(
            tournament_config,
            benchmark,
            event_properties,
        )
        checker = HardConstraintChecker(tournament_config)

        ga_params = app_config.ga_params
        n_total_pop = ga_params.population_size * ga_params.num_islands
        n_objectives = len(evaluator.objectives)
        nsga3 = NSGA3(
            rng=rng,
            n_objectives=n_objectives,
            n_total_pop=n_total_pop,
        )

        selection = RandomSelect(rng)
        operators = app_config.operators
        crossovers = tuple(
            build_crossovers(
                rng,
                operators,
                event_factory,
                event_properties,
            )
        )
        mutations = tuple(
            build_mutations(
                rng,
                operators,
                event_factory,
                event_properties,
            )
        )
        builder = ScheduleBuilder(
            event_factory=event_factory,
            event_properties=event_properties,
            config=tournament_config,
            rng=rng,
        )

        # builder_naive = ScheduleBuilderNaive(
        #     event_factory=event_factory,
        #     event_properties=event_properties,
        #     config=tournament_config,
        #     rng=rng,
        # )
        # naive_schedule = builder_naive.build()
        # repairer.repair(naive_schedule)

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

    def handle_seed_file(self) -> None:
        """Handle the seed file for the genetic algorithm."""
        args = self.app_config.arguments
        config = self.app_config.tournament
        path = Path(args.seed_file).resolve()

        if args.flush and path.exists():
            path.unlink(missing_ok=True)
        path.touch(exist_ok=True)

        if not args.import_file:
            logger.debug("No import file specified, skipping import step.")
            return

        import_path = Path(args.import_file).resolve()
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
            summary_gen.generate(imported_schedule_csv, report_path)

        if not args.add_import_to_population:
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

    def run_preflight_checks(self) -> None:
        """Run preflight checks on the tournament configuration."""
        try:
            self.check_round_definitions()
            self.check_total_capacity()
            self.check_location_time_overlaps()
        except ValueError:
            logger.exception("Preflight checks failed. Please review the configuration.")
        logger.debug("All preflight checks passed successfully.")

    def check_round_definitions(self) -> None:
        """Check that round definitions are valid."""
        c = self.app_config.tournament
        defined_round_types = {r.roundtype for r in c.rounds}
        if diff := defined_round_types.difference(set(c.roundreqs)):
            msg = f"Defined round types {diff} are not required."
            raise ValueError(msg)
        logger.debug("Check passed: All required round types (%s) are defined.", c.roundreqs.keys())

    def check_total_capacity(self) -> None:
        """Check for total available vs. required event slots."""
        c = self.app_config.tournament
        for r in c.rounds:
            rt = r.roundtype
            required = (c.num_teams * r.rounds_per_team) / r.teams_per_round
            available = r.num_timeslots * (len(r.locations) // r.teams_per_round)
            if required > available:
                msg = (
                    f"Capacity impossible for TournamentRound '{rt}':\n"
                    f"  - Required team-events: {required}\n"
                    f"  - Total available team-event slots: {available}\n"
                    f"  - Suggestion: Increase duration, locations, or start/end times for this round."
                )
                raise ValueError(msg)
            logger.debug("Check passed: Capacity sufficient for TournamentRound '%s' - %d/%d.", rt, required, available)
        logger.debug("Check passed: Overall capacity is sufficient.")

    def check_location_time_overlaps(self) -> None:
        """Check if different round types are scheduled in the same locations at the same time."""
        ep = self.event_properties
        booked_slots = defaultdict(list)
        for e in self.event_factory.build_indices():
            loc_str = ep.loc_str[e]
            loc_idx = ep.loc_idx[e]
            timeslot = ep.timeslot[e]
            roundtype = ep.roundtype[e]
            for existing_ts, existing_rt in booked_slots.get(loc_idx, []):
                if timeslot.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: TournamentRound '{roundtype}' and '{existing_rt}' "
                        f"are scheduled in the same location ({loc_str} {loc_idx}) "
                        f"at overlapping times ({timeslot} and "
                        f"{existing_ts})."
                    )
                    raise ValueError(msg)
            booked_slots[loc_idx].append((timeslot, roundtype))
        logger.debug("Check passed: No location/time overlaps found.")
