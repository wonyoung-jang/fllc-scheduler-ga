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
from ..genetic.builder import ScheduleBuilder
from ..genetic.fitness import FitnessEvaluator, HardConstraintChecker
from ..io.ga_exporter import ScheduleSummaryGenerator
from ..io.importer import CsvImporter
from ..operators.crossover import build_crossovers
from ..operators.mutation import build_mutations
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import RandomSelect
from .benchmark import FitnessBenchmark
from .constants import FLUSH, FLUSH_BENCHMARKS, IMPORT_ADD_TO_POP, IMPORT_FILE, SEED_FILE

if TYPE_CHECKING:
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from ..operators.selection import Selection
    from .app_config import AppConfig

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
    max_events_per_team: int

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

        Schedule.set_teams_list(np.arange(tournament_config.num_teams, dtype=int))
        Schedule.set_conflict_matrix(event_factory.build_conflict_matrix())
        Schedule.set_event_properties(event_properties)
        Schedule.set_event_map(event_factory.as_mapping())

        max_events_per_team = sum(r.rounds_per_team for r in tournament_config.rounds)

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
            flush_benchmarks=FLUSH_BENCHMARKS,
        )
        evaluator = FitnessEvaluator(
            tournament_config,
            benchmark,
            event_properties,
            max_events_per_team,
        )
        checker = HardConstraintChecker(tournament_config)

        num_objectives = len(evaluator.objectives)
        params = app_config.ga_params
        nsga3 = NSGA3(
            rng=rng,
            n_objectives=num_objectives,
            n_total_pop=params.population_size * params.num_islands,
        )
        selection = RandomSelect(rng)
        operators = app_config.operators
        crossovers = tuple(build_crossovers(rng, operators, event_factory, event_properties))
        mutations = tuple(build_mutations(rng, operators, event_factory, event_properties))
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
            max_events_per_team=max_events_per_team,
        )

    def handle_seed_file(self) -> None:
        """Handle the seed file for the genetic algorithm."""
        config = self.app_config.tournament
        path = SEED_FILE.resolve()

        if FLUSH and path.exists():
            path.unlink(missing_ok=True)
        path.touch(exist_ok=True)

        if not IMPORT_FILE:
            logger.debug("No import file specified, skipping import step.")
            return

        schedule_csv_path = Path(IMPORT_FILE).resolve()
        csv_import = CsvImporter(schedule_csv_path, config, self.event_factory, self.event_properties)
        csv_schedule = csv_import.schedule
        pop = np.array([csv_schedule.schedule], dtype=int)

        if fits := self.evaluator.evaluate_population(pop):
            sched_fits, team_fits = fits
            csv_schedule.fitness = sched_fits[0]
            csv_schedule.team_fitnesses = team_fits[0]
            csv_import.schedule.fitness = csv_schedule.fitness
            csv_import.schedule.team_fitnesses = csv_schedule.team_fitnesses
            parent_dir = schedule_csv_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            summary_gen = ScheduleSummaryGenerator()
            summary_gen.generate(csv_import.schedule, report_path)

        if not IMPORT_ADD_TO_POP:
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

        if csv_import.schedule not in population:
            population.append(csv_import.schedule)

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
        _ep = self.event_properties
        booked_slots = defaultdict(list)
        for e in self.event_factory.build_indices():
            loc_str = _ep.loc_str[e]
            loc_idx = _ep.loc_idx[e]
            timeslot = _ep.timeslot[e]
            roundtype = _ep.roundtype[e]
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
