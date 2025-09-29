"""Context for the genetic algorithm parts."""

from __future__ import annotations

import itertools
import pickle
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..genetic.ga import GA
from ..io.export import generate_summary_report
from ..io.importer import CsvImporter
from ..observers.observers import LoggingObserver, TqdmObserver

if TYPE_CHECKING:
    from argparse import Namespace

    from ..data_model.event import EventFactory
    from ..data_model.team import TeamFactory
    from ..genetic.fitness import FitnessEvaluator, HardConstraintChecker
    from ..operators.crossover import Crossover
    from ..operators.mutation import Mutation
    from ..operators.nsga3 import NSGA3
    from ..operators.repairer import Repairer
    from ..operators.selection import Selection
    from .app_config import AppConfig

logger = getLogger(__name__)


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    team_factory: TeamFactory
    evaluator: FitnessEvaluator
    checker: HardConstraintChecker
    repairer: Repairer
    nsga3: NSGA3
    selection: Selection
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]
    event_properties: np.ndarray = None
    max_events_per_team: int = 0

    def __post_init__(self) -> None:
        """Post-initialization to validate the GA context."""
        self.max_events_per_team = sum(r.rounds_per_team for r in self.app_config.tournament.rounds)
        self.event_properties = self.build_event_properties()
        self.run_preflight_checks()

    def build_event_properties(self) -> np.ndarray:
        """Build a numpy array of event properties for fast access during evaluation."""
        num_events = self.app_config.tournament.total_slots_possible
        event_map = self.event_factory.as_mapping()
        round_to_int = self.app_config.tournament.round_to_int

        # Shape: (num_events, num_properties)
        event_props = np.zeros((num_events, 6), dtype=int)
        for i in range(num_events):
            event = event_map[i]
            event_props[i] = [
                int(event.timeslot.start.timestamp()),
                int(event.timeslot.stop.timestamp()),
                event.location.idx,
                event.location.side,
                round_to_int[event.roundtype],
                event.paired.idx if event.paired else -1,
            ]
        logger.debug("Event properties array: %s", event_props)
        return event_props

    def create_ga_instance(self, args: Namespace) -> GA:
        """Create and return a GA instance with the provided configuration."""
        self.handle_seed_file(args)
        return GA(
            context=self,
            rng=self.app_config.rng,
            observers=(TqdmObserver(), LoggingObserver()),
            seed_file=Path(args.seed_file) if args.seed_file else None,
            save_front_only=args.front_only,
        )

    def handle_seed_file(self, args: Namespace) -> None:
        """Handle the seed file for the genetic algorithm."""
        config = self.app_config.tournament
        path = Path(args.seed_file).resolve()

        if args.flush and path.exists():
            path.unlink(missing_ok=True)
        path.touch(exist_ok=True)

        if not args.import_file:
            logger.debug("No import file specified, skipping import step.")
            return

        schedule_csv_path = Path(args.import_file).resolve()
        csv_import = CsvImporter(schedule_csv_path, config, self.event_factory)
        csv_schedule = csv_import.schedule
        pop = np.array([s.to_array(self) for s in [csv_schedule]])

        if fits := self.evaluator.evaluate_population(pop):
            csv_schedule.fitness = fits[0]
            csv_import.schedule.fitness = csv_schedule.fitness
            parent_dir = schedule_csv_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            report_path = parent_dir / "report.txt"
            generate_summary_report(
                csv_import.schedule,
                report_path,
            )

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
        if diff := defined_round_types.difference(set(c.round_requirements)):
            msg = f"Defined round types {diff} are not required."
            raise ValueError(msg)
        logger.debug("Check passed: All required round types (%s) are defined.", c.round_requirements.keys())

    def check_total_capacity(self) -> None:
        """Check for total available vs. required event slots."""
        c = self.app_config.tournament
        for r in c.rounds:
            rt = r.roundtype
            required = (c.num_teams * r.rounds_per_team) / r.teams_per_round
            available = r.num_timeslots * (len(r.locations) // r.teams_per_round)
            if required > available:
                msg = (
                    f"Capacity impossible for Round '{rt}':\n"
                    f"  - Required team-events: {required}\n"
                    f"  - Total available team-event slots: {available}\n"
                    f"  - Suggestion: Increase duration, locations, or start/end times for this round."
                )
                raise ValueError(msg)
            logger.debug("Check passed: Capacity sufficient for Round '%s' - %d/%d.", rt, required, available)
        logger.debug("Check passed: Overall capacity is sufficient.")

    def check_location_time_overlaps(self) -> None:
        """Check if different round types are scheduled in the same locations at the same time."""
        c = self.app_config.tournament
        ef = self.event_factory
        booked_slots = defaultdict(list)
        event_idx_iter = itertools.count()
        for r in c.rounds:
            for e in ef.create_events(r, event_idx_iter):
                loc_key = (type(e.location), e.location)
                for existing_ts, existing_rt in booked_slots.get(loc_key, []):
                    if e.timeslot.overlaps(existing_ts):
                        msg = (
                            f"Configuration conflict: Round '{r.roundtype}' and '{existing_rt}' "
                            f"are scheduled in the same location ({loc_key[0].__name__} {loc_key[1]}) "
                            f"at overlapping times ({e.timeslot} and "
                            f"{existing_ts})."
                        )
                        raise ValueError(msg)
                booked_slots[loc_key].append((e.timeslot, r.roundtype))
        logger.debug("Check passed: No location/time overlaps found.")
