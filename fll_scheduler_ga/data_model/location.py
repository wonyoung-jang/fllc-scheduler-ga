"""Location data model for FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.constants import ASCII_OFFSET


@dataclass(slots=True, frozen=True)
class Location:
    """Data model for a location in the FLL Scheduler GA."""

    name: str
    identity: int | str
    teams_per_round: int
    side: int = field(default=0)

    def __str__(self) -> str:
        """Represent the Location as a string."""
        if self.side:
            if isinstance(self.identity, int):
                return f"{self.name} {chr(ASCII_OFFSET + self.identity)}{self.side}"
            if isinstance(self.identity, str):
                return f"{self.name} {self.identity}{self.side}"

        return f"{self.name} {self.identity}"

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        if self.side:
            return hash((self.identity, self.side))
        return hash(self.identity)
