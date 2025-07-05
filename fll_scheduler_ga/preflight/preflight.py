"""Preflight checks for the fll-scheduler-ga application."""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..config.config import TournamentConfig
from ..data_model.event import EventFactory

if TYPE_CHECKING:
    from ..data_model.event import Event

logger = logging.getLogger(__name__)


def run_preflight_checks(config: TournamentConfig) -> None:
    """Run preflight checks on the tournament configuration."""
    logger.info("Running preflight checks on the tournament configuration")
    _check_round_definitions(config)
    _check_total_capacity(config)
    _check_peak_capacity_bottlenecks(config)
    _check_per_team_feasibility(config)
    _check_location_time_overlaps(config)
    logger.info("All preflight checks passed successfully.")


def _check_round_definitions(config: TournamentConfig) -> None:
    """Check that round definitions are valid."""
    defined_round_types = {r.round_type for r in config.rounds}
    for req_type in config.round_requirements:
        if req_type not in defined_round_types:
            msg = f"Required round type '{req_type}' is not defined in the configuration."
            raise ValueError(msg)
    logger.info("Check passed: All required round types are defined.")


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
        logger.info("Check passed: Capacity sufficient for Round '%s' - %d/%d.", r_type, req_count, available[r_type])
    logger.info("Check passed: Overall capacity is sufficient.")


def _check_peak_capacity_bottlenecks(config: TournamentConfig) -> None:
    """Identify if any single hour is impossibly over-scheduled."""
    event_factory = EventFactory(config)
    all_events: list[Event] = []
    for r_config in config.rounds:
        all_events.extend(event_factory.create_events(r_config))

    slots_by_hour_and_type = defaultdict(lambda: defaultdict(int))
    for event in all_events:
        hour_key = event.time_slot.start.strftime("%H:00")
        round_config = next(r for r in config.rounds if r.round_type == event.round_type)
        slots_by_hour_and_type[event.round_type][hour_key] += round_config.teams_per_round

    for r_type, hourly_slots in slots_by_hour_and_type.items():
        required_rounds_per_team = config.round_requirements.get(r_type, 0)
        total_hours = len(hourly_slots)
        if total_hours == 0:
            continue
        avg_teams_per_hour_needed = (config.num_teams * required_rounds_per_team) / total_hours

        for hour, capacity in hourly_slots.items():
            if capacity < avg_teams_per_hour_needed:
                msg = (
                    f"Potential Bottleneck Detected in Round '{r_type}' at {hour}:\n"
                    f"  - This hour only has capacity for {capacity} teams.\n"
                    f"  - The average demand is ~{avg_teams_per_hour_needed:.1f} teams per hour.\n"
                    f"  - This may create scheduling pressure or result in less optimal schedules."
                )
                logger.warning(msg)
    logger.info("Check passed: No obvious hourly bottlenecks found.")


def _check_per_team_feasibility(config: TournamentConfig) -> None:
    """Check if a single team has enough time slots in the day for all its events."""
    all_times = [t for r in config.rounds for t in [r.start_time]]
    if not all_times:
        return

    all_slots = []
    for r_config in config.rounds:
        start_dt = datetime.strptime(r_config.start_time, "%H:%M").replace(tzinfo=UTC)
        for i in range(r_config.num_slots):
            slot_start = start_dt + (i * r_config.duration_minutes)
            all_slots.append(slot_start)

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
    logger.info("Check passed: Per-team event load seems feasible.")


def _check_location_time_overlaps(config: TournamentConfig) -> None:
    """Check if different round types are scheduled in the same locations at the same time."""
    booked_slots = defaultdict(list)
    event_factory = EventFactory(config)

    for r_config in config.rounds:
        for event in event_factory.create_events(r_config):
            location_key = (type(event.location), event.location)
            for existing_ts, existing_r_type in booked_slots.get(location_key, []):
                if event.time_slot.overlaps(existing_ts):
                    msg = (
                        f"Configuration conflict: Round '{r_config.round_type}' and '{existing_r_type}' "
                        f"are scheduled in the same location ({location_key[0].__name__} {location_key[1]}) "
                        f"at overlapping times ({event.time_slot.start_str}-{event.time_slot.stop_str} and "
                        f"{existing_ts.start_str}-{existing_ts.stop_str})."
                    )
                    raise ValueError(msg)
            booked_slots[location_key].append((event.time_slot, r_config.round_type))
    logger.info("Check passed: No location/time overlaps found.")
