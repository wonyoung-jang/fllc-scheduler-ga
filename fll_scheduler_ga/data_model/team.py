"""Team data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = getLogger(__name__)


@dataclass(slots=True)
class TeamFactory:
    """Factory class to create Team indices."""

    teams: np.ndarray[int]
