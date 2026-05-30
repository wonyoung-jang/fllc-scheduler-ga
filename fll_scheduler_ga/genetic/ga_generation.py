"""Generation tracker for GA."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GaGeneration:
    """Class for tracking GA generation information."""

    curr: int = 0
