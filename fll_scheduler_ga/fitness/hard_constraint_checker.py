"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)


@dataclass(slots=True)
class HardConstraintChecker:
    """Validates hard constraints for a schedule."""

    constraints: tuple[HardConstraint, ...]

    def check(self, schedule: Schedule) -> bool:
        """Check the hard constraints of a schedule."""
        return not any(constraint.is_violated(schedule) for constraint in self.constraints)


@dataclass(slots=True)
class HardConstraint(ABC):
    """Base class for hard constraints."""

    @abstractmethod
    def is_violated(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""


@dataclass(slots=True)
class HardConstraintTruthiness(HardConstraint):
    """A hard constraint that always returns True."""

    def is_violated(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return not schedule


@dataclass(slots=True)
class HardConstraintSize(HardConstraint):
    """A hard constraint that checks the size of the schedule."""

    total_slots_required: int

    def is_violated(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return schedule.get_size() != self.total_slots_required


@dataclass(slots=True)
class HardConstraintNoRoundsNeeded(HardConstraint):
    """A hard constraint that checks if any rounds are still needed."""

    def is_violated(self, schedule: Schedule) -> bool:
        """Check the hard constraint on the given schedule."""
        return schedule.any_rounds_needed()
