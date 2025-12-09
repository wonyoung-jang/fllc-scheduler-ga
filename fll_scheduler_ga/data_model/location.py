"""Location data model for FLL Scheduler GA."""

from pydantic import BaseModel, Field

from ..config.constants import ASCII_OFFSET


class Location(BaseModel):
    """Data model for a location in the FLL Scheduler GA."""

    model_config = {"frozen": True}
    idx: int = Field(ge=0)
    locationtype: str = Field(min_length=1)
    name: int = Field(ge=1)
    side: int = Field(ge=-1)
    teams_per_round: int = Field(ge=1)

    def __str__(self) -> str:
        """Represent the Location as a string."""
        ltr_id = chr(ASCII_OFFSET + self.name)

        if self.side > 0:
            return f"{self.locationtype} {ltr_id}{self.side}"
        return f"{self.locationtype} {ltr_id}"

    def __hash__(self) -> int:
        """Hash the Room based on its identity."""
        return hash((self.name, self.side))
