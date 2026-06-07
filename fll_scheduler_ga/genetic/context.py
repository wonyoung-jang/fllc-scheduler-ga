"""Context for the genetic algorithm parts."""

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

from fll_scheduler_ga.domain.model import Schedule

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from fll_scheduler_ga.domain.model import EventProperties, EventRepository
    from fll_scheduler_ga.genetic.fitness import FitnessEvaluator
    from fll_scheduler_ga.genetic.operator import NSGA3, Crossover, Mutation, Repairer, Selection

logger = getLogger(__name__)


@dataclass(slots=True)
class ScheduleBuilder:
    """Builder for building a valid random schedule."""

    evt_prop: EventProperties
    rng: np.random.Generator
    round_idx_to_tpr: dict[int, int]
    roundtype_events: dict[int, list[int]]

    def build(self) -> Schedule:
        """Construct and return the final schedule."""
        s = Schedule(origin="Builder")
        for ri, evts in self.roundtype_events.items():
            if self.round_idx_to_tpr[ri] == 1:
                self.build_singles(s, self.rng.permutation(evts), ri)
            elif self.round_idx_to_tpr[ri] == 2:
                self.build_matches(s, self.rng.permutation(evts), ri)
        return s

    def build_singles(self, s: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all judging events for a specific round type."""
        for event in events:
            shuffled_teams = self.rng.permutation(s.all_rounds_needed(roundtype))
            available = (t for t in shuffled_teams if not s.conflicts(t, event))
            if (team := next(available, None)) is not None:
                s.assign(team, event)

    def build_matches(self, s: Schedule, events: np.ndarray, roundtype: int) -> None:
        """Book all events for a specific round type."""
        loc_sides_where_1 = self.evt_prop.loc_side[events] == 1
        side1s = events[loc_sides_where_1.nonzero()[0]]
        side2s = self.evt_prop.paired_idx[side1s]
        for e1, e2 in zip(side1s, side2s, strict=True):
            shuffled_teams = self.rng.permutation(s.all_rounds_needed(roundtype))
            available = (t for t in shuffled_teams if not s.conflicts(t, e1))
            if (t1 := next(available, None)) is not None:
                s.assign(t1, e1)
            if (t2 := next(available, None)) is not None:
                s.assign(t2, e2)


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    evt_repo: EventRepository
    evt_prop: EventProperties
    evaluator: FitnessEvaluator
    checker: Callable[[Schedule], bool]
    builder: ScheduleBuilder
    repairer: Repairer
    nsga3: NSGA3
    selection: Selection
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]

    def build(self) -> Schedule:
        """Build a new schedule using the builder."""
        return self.builder.build()

    def check(self, schedule: Schedule) -> bool:
        """Check a schedule using the hard constraint checker."""
        return self.checker(schedule)

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
