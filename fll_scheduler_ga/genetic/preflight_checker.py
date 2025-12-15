"""Context for the genetic algorithm parts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.timeslot import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class PreFlightChecker:
    """Run pre-flight checks on the tournament configuration."""

    event_properties: EventProperties
    event_factory: EventFactory

    @classmethod
    def build_then_run(cls, event_properties: EventProperties, event_factory: EventFactory) -> None:
        """Build a PreFlightChecker and run all checks."""
        checker = cls(event_properties=event_properties, event_factory=event_factory)
        checker.run_checks()

    def run_checks(self) -> None:
        """Run all pre-flight checks."""
        try:
            self.check_location_time_overlaps()
        except ValueError:
            logger.exception("Preflight checks failed. Please review the configuration.")
        logger.debug("All preflight checks passed successfully.")

    def check_location_time_overlaps(self) -> None:
        """Check if different round types are scheduled in the same locations at the same time."""
        ep = self.event_properties
        booked_slots: dict[int, list[tuple[TimeSlot, str]]] = defaultdict(list)
        for e in self.event_factory.build_indices():
            loc_str = ep.loc_str[e]
            loc_idx = ep.loc_idx[e]
            ts = ep.timeslot[e]
            rt = ep.roundtype[e]
            for existing_ts, existing_rt in booked_slots.get(loc_idx, []):
                if ts.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: TournamentRound '{rt}' and '{existing_rt}' "
                        f"are scheduled in the same location ({loc_str} {loc_idx}) "
                        f"at overlapping times ({ts} and "
                        f"{existing_ts})."
                    )
                    raise ValueError(msg)
            booked_slots[loc_idx].append((ts, rt))
        logger.debug("Check passed: No location/time overlaps found.")
