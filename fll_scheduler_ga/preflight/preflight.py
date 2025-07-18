"""Preflight checks for the fll-scheduler-ga application."""

import logging
from collections import defaultdict

from ..config.config import TournamentConfig
from ..data_model.event import EventFactory

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
    defined_round_types = {r.round_type for r in config.rounds}
    for req_type in config.round_requirements:
        if req_type not in defined_round_types:
            msg = f"Required round type '{req_type}' is not defined in the configuration."
            raise ValueError(msg)
    logger.debug("Check passed: All required round types are defined.")


def _check_total_capacity(config: TournamentConfig) -> None:
    """Check for total available vs. required event slots."""
    required = defaultdict(int)
    for r_config in config.rounds:
        num_teams_for_round = config.num_teams * r_config.rounds_per_team
        required[r_config.round_type] += num_teams_for_round / r_config.teams_per_round

    available = defaultdict(int)
    for r_config in config.rounds:
        available[r_config.round_type] += r_config.num_slots * r_config.num_locations

    for r_type, req_count in required.items():
        if req_count > available[r_type]:
            msg = (
                f"Capacity impossible for Round '{r_type}':\n"
                f"  - Required team-events: {req_count}\n"
                f"  - Total available team-event slots: {available[r_type]}\n"
                f"  - Suggestion: Increase duration, locations, or start/end times for this round."
            )
            raise ValueError(msg)
        logger.debug("Check passed: Capacity sufficient for Round '%s' - %d/%d.", r_type, req_count, available[r_type])
    logger.debug("Check passed: Overall capacity is sufficient.")


def _check_per_team_feasibility(config: TournamentConfig) -> None:
    """Check if a single team has enough time slots in the day for all its events."""
    all_times = [t for r in config.rounds for t in [r.start_time]]
    if not all_times:
        return

    all_slots = []
    for r_config in config.rounds:
        start_dt = r_config.start_time
        for i in range(r_config.num_slots):
            slot_start = start_dt + (i * r_config.duration_minutes)
            all_slots.append(slot_start)

        if r_config.stop_time:
            stop_dt = r_config.stop_time
            if all_slots[-1] + r_config.duration_minutes > stop_dt:
                msg = (
                    f"Round '{r_config.round_type}' exceeds configured stop time.\n"
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

    for r_config in config.rounds:
        for event in event_factory.create_events(r_config):
            location_key = (type(event.location), event.location)
            for existing_ts, existing_r_type in booked_slots.get(location_key, []):
                if event.timeslot.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: Round '{r_config.round_type}' and '{existing_r_type}' "
                        f"are scheduled in the same location ({location_key[0].__name__} {location_key[1]}) "
                        f"at overlapping times ({event.timeslot.start_str}-{event.timeslot.stop_str} and "
                        f"{existing_ts.start_str}-{existing_ts.stop_str})."
                    )
                    raise ValueError(msg)
            booked_slots[location_key].append((event.timeslot, r_config.round_type))
    logger.debug("Check passed: No location/time overlaps found.")
