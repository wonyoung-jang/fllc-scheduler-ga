"""Location data model for FLL Scheduler GA."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config.constants import ASCII_OFFSET

_RE_TABLE = re.compile(r"Table ([A-Z])(\d)")


@dataclass(slots=True, frozen=True)
class Location(ABC):
    """Data model for a location in the FLL Scheduler GA."""

    identity: int
    teams_per_round: int

    @classmethod
    @abstractmethod
    def from_string(cls, location_str: str, teams_per_round: int) -> "Location | None":
        """Attempt to parse a location from a string."""
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class Room(Location):
    """Data model for a room in the FLL Scheduler GA."""

    def __str__(self) -> str:
        """Represent the Room as a string."""
        return f"{self.__class__.__name__} {self.identity}"

    @classmethod
    def from_string(cls, location_str: str, teams_per_round: int) -> "Room | None":
        """Attempt to parse a room from a string."""
        if location_str.startswith("Room"):
            try:
                room_id = int(location_str.split(" ")[1])
                return cls(identity=room_id, teams_per_round=teams_per_round)
            except (ValueError, IndexError):
                return None
        return None


@dataclass(slots=True, frozen=True)
class Table(Location):
    """Data model for a table in the FLL Scheduler GA."""

    side: int

    def __str__(self) -> str:
        """Represent the Table as a string."""
        return f"{self.__class__.__name__} {chr(ASCII_OFFSET + self.identity)}{self.side}"

    def __hash__(self) -> int:
        """Hash the Table based on its identity and side."""
        return hash((self.identity, self.side))

    @classmethod
    def from_string(cls, location_str: str, teams_per_round: int) -> "Table | None":
        """Attempt to parse a table from a string."""
        match = _RE_TABLE.match(location_str)
        if match:
            table_char, side = match.groups()
            table_id = ord(table_char) - ASCII_OFFSET
            return cls(identity=table_id, side=int(side), teams_per_round=teams_per_round)
        return None


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
        msg = (
            f"Unsupported number of teams per round: {teams_per_round}. Only {list(location_map.keys())} are supported."
        )
        raise ValueError(msg)

    return location_map[teams_per_round]


def get_location_type_from_string(location_str: str, teams_per_round: int) -> Location | None:
    """Parse a location string from the CSV header into a Location object."""
    if not location_str:
        return None

    locations = (
        Room,
        Table,
    )

    for loc_type in locations:
        if loc := loc_type.from_string(location_str, teams_per_round):
            return loc

    return None
