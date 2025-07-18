"""Location data model for FLL Scheduler GA."""

from abc import ABC
from dataclasses import dataclass, field

ASCII_OFFSET = 64


@dataclass(slots=True, frozen=True)
class Location(ABC):
    """Data model for a location in the FLL Scheduler GA."""

    identity: int = field(hash=True)
    teams_per_round: int


@dataclass(slots=True, frozen=True)
class Room(Location):
    """Data model for a room in the FLL Scheduler GA."""

    def __str__(self) -> str:
        """Represent the Room as a string."""
        return f"{self.__class__.__name__} {self.identity}"


@dataclass(slots=True, frozen=True)
class Table(Location):
    """Data model for a table in the FLL Scheduler GA."""

    side: int = field(hash=True)

    def __str__(self) -> str:
        """Represent the Table as a string."""
        return f"{self.__class__.__name__} {chr(ASCII_OFFSET + self.identity)}{self.side}"


def get_location_type(teams_per_round: int) -> Room | Table:
    """Get the location type based on the teams per round.

    Args:
        teams_per_round (int): The number of teams per round.

    Returns:
        Room | Table: The corresponding location type for the round.

    """
    location_map = {
        1: Room,
        2: Table,
    }

    if teams_per_round not in location_map:
        msg = f"Unsupported number of teams per round: {teams_per_round}. Only 1 or 2 are supported."
        raise ValueError(msg)

    return location_map[teams_per_round]
