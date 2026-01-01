"""Location data model for FLL Scheduler GA."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.constants import ASCII_OFFSET

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ..config.pydantic_schemas import LocationModel


@dataclass(slots=True)
class LocationModelsParser:
    """Parser for location models into Location instances."""

    models: Iterable[LocationModel]

    def parse(self) -> tuple[Location, ...]:
        """Parse location models into Location instances."""
        _idx_counter = itertools.count()

        def _generate_locations() -> Iterator[Location]:
            for loctype in self.models:
                for name in range(1, loctype.count + 1):
                    for side_iter in range(1, loctype.sides + 1):
                        yield Location(
                            idx=next(_idx_counter),
                            locationtype=loctype.name,
                            name=name,
                            side=-1 if loctype.sides == 1 else side_iter,
                            teams_per_round=loctype.sides,
                        )

        locations = tuple(_generate_locations())
        if not locations:
            msg = "No locations defined in the configuration file."
            raise ValueError(msg)

        return locations


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
        self._str = f"{self.locationtype} {ltr_id}"

        self._hash = hash((self.name, self.side))

    def __str__(self) -> str:
        """Represent the Location as a string."""
        return self._str

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        return self._hash
