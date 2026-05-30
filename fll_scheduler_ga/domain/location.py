"""Location data model for FLL Scheduler GA."""

from dataclasses import dataclass

ASCII_OFFSET = 64


@dataclass(slots=True)
class Location:
    """Data model for a location in the FLL Scheduler GA."""

    idx: int = 0
    locationtype: str = "Null"
    name: int = 1
    side: int = -1
    teams_per_round: int = 1
    _str: str = ""
    _hash: int = 0

    def __post_init__(self) -> None:
        """Post-initialization to set private attributes."""
        ltr_id = chr(ASCII_OFFSET + self.name)
        if self.side > 0:
            self._str = f"{self.locationtype} {ltr_id}{self.side}"
        else:
            self._str = f"{self.locationtype} {ltr_id}"
        self._hash = hash((self.name, self.side))

    def __str__(self) -> str:
        """Represent the Location as a string."""
        return self._str

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        return self._hash
