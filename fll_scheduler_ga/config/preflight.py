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
        _check_per_team_feasibility(config)
        _check_location_time_overlaps(config, event_factory)
    except ValueError:
        logger.exception("Preflight checks failed. Please review the configuration.")
    logger.info("All preflight checks passed successfully.")


def _check_round_definitions(config: TournamentConfig) -> None:
    """Check that round definitions are valid."""
    defined_round_types = {r.roundtype for r in config.rounds}
    for req_type in config.round_requirements:
        if req_type not in defined_round_types:
            msg = f"Required round type '{req_type}' is not defined in the configuration."
            raise ValueError(msg)
    logger.debug("Check passed: All required round types are defined.")


def _check_total_capacity(config: TournamentConfig) -> None:
    """Check for total available vs. required event slots."""
    required = {}
    available = {}
    for r in config.rounds:
        required[r.roundtype] = (config.num_teams * r.rounds_per_team) / r.teams_per_round
        available[r.roundtype] = r.get_num_slots() * r.num_locations

    for rt, count in required.items():
        if count > available[rt]:
            msg = (
                f"Capacity impossible for Round '{rt}':\n"
                f"  - Required team-events: {count}\n"
                f"  - Total available team-event slots: {available[rt]}\n"
                f"  - Suggestion: Increase duration, locations, or start/end times for this round."
            )
            raise ValueError(msg)
        logger.debug("Check passed: Capacity sufficient for Round '%s' - %d/%d.", rt, count, available[rt])
    logger.debug("Check passed: Overall capacity is sufficient.")


def _check_per_team_feasibility(config: TournamentConfig) -> None:
    """Check if a single team has enough time slots in the day for all its events."""
    if not [r.start_time for r in config.rounds]:
        return

    all_slots = []
    for r in config.rounds:
        start_dt = r.start_time
        for i in range(r.get_num_slots()):
            slot_start = start_dt + (i * r.duration_minutes)
            all_slots.append(slot_start)

        if r.stop_time:
            stop_dt = r.stop_time
            if all_slots[-1] + r.duration_minutes > stop_dt:
                msg = (
                    f"Round '{r.roundtype}' exceeds configured stop time.\n"
                    f"  - Last slot starts at {all_slots[-1]} but should not exceed {stop_dt}."
                )
                raise ValueError(msg)

    total_time_slots_available = len(set(all_slots))
    total_rounds_per_team = sum(config.round_requirements.values())

    if total_rounds_per_team > total_time_slots_available:
        msg = (
            "Feasibility check failed: A single team is required to participate in more "
            "rounds than there are available time slots in the day.\n"
            f"  - Total rounds required per team: {total_rounds_per_team}\n"
            f"  - Total unique time slots available in the tournament: {total_time_slots_available}\n"
            f"  - Suggestion: Add more time slots by extending the day or "
            "reduce the number of rounds required per team."
        )
        raise ValueError(msg)
    logger.debug("Check passed: Per-team event load seems feasible.")


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
