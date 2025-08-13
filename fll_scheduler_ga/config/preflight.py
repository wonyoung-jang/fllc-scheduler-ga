"""Preflight checks for the fll-scheduler-ga application."""

import logging
from collections import defaultdict

from ..data_model.event import EventFactory
from .config import TournamentConfig

logger = logging.getLogger(__name__)


def run_preflight_checks(config: TournamentConfig, event_factory: EventFactory) -> None:
    """Run preflight checks on the tournament configuration."""
    try:
        _check_round_definitions(config)
        _check_total_capacity(config)
        _check_location_time_overlaps(config, event_factory)
    except ValueError:
        logger.exception("Preflight checks failed. Please review the configuration.")
    logger.debug("All preflight checks passed successfully.")


def _check_round_definitions(config: TournamentConfig) -> None:
    """Check that round definitions are valid."""
    defined_round_types = {r.roundtype for r in config.rounds}
    if diff := defined_round_types.difference(set(config.round_requirements)):
        msg = f"Defined round types {diff} are not required."
        raise ValueError(msg)
    logger.debug("Check passed: All required round types (%s) are defined.", config.round_requirements.keys())


def _check_total_capacity(config: TournamentConfig) -> None:
    """Check for total available vs. required event slots."""
    for r in config.rounds:
        rt = r.roundtype
        required = (config.num_teams * r.rounds_per_team) / r.teams_per_round
        available = r.get_num_slots() * (len(r.locations) // r.teams_per_round)
        if required > available:
            msg = (
                f"Capacity impossible for Round '{rt}':\n"
                f"  - Required team-events: {required}\n"
                f"  - Total available team-event slots: {available}\n"
                f"  - Suggestion: Increase duration, locations, or start/end times for this round."
            )
            raise ValueError(msg)
        logger.debug("Check passed: Capacity sufficient for Round '%s' - %d/%d.", rt, required, available)
    logger.debug("Check passed: Overall capacity is sufficient.")


def _check_location_time_overlaps(config: TournamentConfig, event_factory: EventFactory) -> None:
    """Check if different round types are scheduled in the same locations at the same time."""
    booked_slots = defaultdict(list)
    for r in config.rounds:
        for e in event_factory.create_events(r):
            loc_key = (type(e.location), e.location)
            for existing_ts, existing_rt in booked_slots.get(loc_key, []):
                if e.timeslot.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: Round '{r.roundtype}' and '{existing_rt}' "
                        f"are scheduled in the same location ({loc_key[0].__name__} {loc_key[1]}) "
                        f"at overlapping times ({e.timeslot} and "
                        f"{existing_ts})."
                    )
                    raise ValueError(msg)
            booked_slots[loc_key].append((e.timeslot, r.roundtype))
    logger.debug("Check passed: No location/time overlaps found.")
