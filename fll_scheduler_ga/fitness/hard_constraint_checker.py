"""Fitness evaluator for the FLL Scheduler GA."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fll_scheduler_ga.data_model.schedule import Schedule


@dataclass(slots=True)
class HardConstraintChecker:
    """Validates hard constraints for a schedule."""

    constraints: tuple[HardConstraint, ...]

    def check(self, schedule: Schedule) -> bool:
        """Check the hard constraints of a schedule."""
        return not any(constraint(schedule) for constraint in self.constraints)


@dataclass(slots=True)
class HardConstraint(ABC):
    """Base class for hard constraints."""

    @abstractmethod
    def __call__(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""


@dataclass(slots=True)
class HardConstraintTruthiness(HardConstraint):
    """A hard constraint that always returns True."""

    def __call__(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return not schedule


@dataclass(slots=True)
class HardConstraintSize(HardConstraint):
    """A hard constraint that checks the size of the schedule."""

    total_slots_required: int

    def __call__(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return schedule.get_size() != self.total_slots_required


@dataclass(slots=True)
class HardConstraintNoRoundsNeeded(HardConstraint):
    """A hard constraint that checks if any rounds are still needed."""

    def __call__(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return schedule.any_rounds_needed()
