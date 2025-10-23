"""Location data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.constants import ASCII_OFFSET


@dataclass(slots=True, frozen=True)
class Location:
    """Data model for a location in the FLL Scheduler GA."""

    idx: int
    locationtype: str
    name: int
    side: int
    teams_per_round: int

    def __str__(self) -> str:
        """Represent the Location as a string."""
        ltr_id = f"{chr(ASCII_OFFSET + self.name)}"

        if self.side > 0:
            return f"{self.locationtype} {ltr_id}{self.side}"
        return f"{self.locationtype} {ltr_id}"

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        if self.side:
            return hash((self.name, self.side))
        return hash(self.name)
