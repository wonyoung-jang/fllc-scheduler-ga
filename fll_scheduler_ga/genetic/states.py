"""Module defining states for the Genetic Algorithm using the State design pattern."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ga import GA


@dataclass(slots=True)
class GaState(ABC):
    """Abstract base class for GA states."""

    _context: "GA"

    @abstractmethod
    def run(self) -> None:
        """Execute the logic for this state."""


@dataclass(slots=True)
class InitializingState(GaState):
    """State representing the initialization phase of the GA."""

    def run(self) -> None:
        """Initialize the GA and transition to the evolving state."""
        self._context.logger.info("GA is in Initializing state.")
        self._context.initialize_population()
        self._context.setstate(EvolvingState(self._context))


@dataclass(slots=True)
class EvolvingState(GaState):
    """State representing the evolving phase of the GA."""

    def run(self) -> None:
        """Run the evolution process for a set number of generations."""
        self._context.logger.info("GA is in Evolving state.")
        self._context.generation()
        self._context.setstate(TerminatedState(self._context))


@dataclass(slots=True)
class TerminatedState(GaState):
    """State representing the termination phase of the GA."""

    def run(self) -> None:
        """Finalize the GA and log the final summary."""
        self._context.logger.info("GA has terminated.")
        self._context.log_final_summary()
