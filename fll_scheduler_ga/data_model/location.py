"""Location data model for FLL Scheduler GA."""

from dataclasses import dataclass, field

from ..config.constants import ASCII_OFFSET, RE_TABLE, FllcLocation


@dataclass(slots=True, frozen=True)
class Location:
    """Data model for a location in the FLL Scheduler GA."""

    name: str
    identity: int
    teams_per_round: int
    side: int = field(default=0)

    def __str__(self) -> str:
        """Represent the Location as a string."""
        if self.side:
            return f"{self.name} {chr(ASCII_OFFSET + self.identity)}{self.side}"
        return f"{self.name} {self.identity}"

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        if self.side:
            return hash((self.identity, self.side))
        return hash(self.identity)

    @classmethod
    def from_string(cls, location_str: str, teams_per_round: int) -> "Location | None":
        """Attempt to parse a location from a string."""
        if location_str.startswith(FllcLocation.ROOM):
            room_id = int(location_str.split(" ")[1])
            return cls(
                name=FllcLocation.ROOM,
                identity=room_id,
                teams_per_round=teams_per_round,
            )

        if match := RE_TABLE.match(location_str):
            table_char, side = match.groups()
            table_id = ord(table_char) - ASCII_OFFSET
            return cls(
                name=FllcLocation.TABLE,
                identity=table_id,
                teams_per_round=teams_per_round,
                side=int(side),
            )

        return None
