"""Generation tracker for GA."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GaGeneration:
    """Class for tracking GA generation information."""

    curr: int

    def increment(self) -> None:
        """Increment the generation counter."""
        self.curr += 1
