"""Fixtures for testing fll_scheduler_ga package."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fll_scheduler_ga.config.app_config import AppConfig
from fll_scheduler_ga.data_model.app_schemas import TournamentConfig
from fll_scheduler_ga.data_model.event import EventFactory, EventProperties
from fll_scheduler_ga.data_model.schedule import Schedule, ScheduleContext
from fll_scheduler_ga.data_model.timeslot import TIME_FORMAT_MAP, TimeSlot, parse_time_str

FMT_24H: str = TIME_FORMAT_MAP[24]
FMT_12H: str = TIME_FORMAT_MAP[12]


@pytest.fixture
def minimal_config_dict() -> dict[str, Any]:
    """Return a minimal valid configuration dictionary."""
    return {
        "genetic": {
            "parameters": {
                "population_size": 4,
                "generations": 2,
            },
            "operator": {
                "crossover": {
                    "types": ["KPoint"],
                    "k_vals": [1],
                },
                "mutation": {
                    "types": ["SwapTeam_CrossTimeLocation"],
                },
            },
            "stagnation": {
                "enable": False,
            },
        },
        "runtime": {
            "seed_file": "test_seed.pkl",
        },
        "io": {
            "imports": {},
            "exports": {},
        },
        "fitness": {},
        "tournament": {
            "teams": 4,
            "locations": [
                {
                    "name": "Room",
                    "count": 1,
                    "sides": 1,
                },
                {
                    "name": "Table",
                    "count": 2,
                    "sides": 2,
                },
            ],
            "rounds": [
                {
                    "roundtype": "Judging",
                    "location": "Room",
                    "rounds_per_team": 1,
                    "teams_per_round": 1,
                    "start_time": "09:00",
                    "stop_time": "09:30",
                    "duration_cycle": 5,
                    "duration_active": 3,
                },
                {
                    "roundtype": "Table",
                    "location": "Table",
                    "rounds_per_team": 1,
                    "teams_per_round": 2,
                    "start_time": "12:00",
                    "stop_time": "12:30",
                    "duration_cycle": 5,
                    "duration_active": 3,
                },
            ],
        },
    }


@pytest.fixture
def app_config(minimal_config_dict: dict[str, Any], tmp_path: Path) -> AppConfig:
    """Create an AppConfig instance from a temporary file."""
    config_file = tmp_path / "config.json"
    with config_file.open("w") as f:
        json.dump(minimal_config_dict, f)
    return AppConfig.build(config_file)


@pytest.fixture
def tournament_config(app_config: AppConfig) -> TournamentConfig:
    """Return the TournamentConfig."""
    return app_config.tournament


@pytest.fixture
def event_factory(tournament_config: TournamentConfig) -> EventFactory:
    """Return an EventFactory."""
    return EventFactory(tournament_config)


@pytest.fixture
def event_properties(tournament_config: TournamentConfig, event_factory: EventFactory) -> EventProperties:
    """Return EventProperties."""
    return EventProperties.build(tournament_config.get_n_total_events(), event_factory.as_mapping())


@pytest.fixture
def schedule_context(
    tournament_config: TournamentConfig,
    event_factory: EventFactory,
    event_properties: EventProperties,
) -> ScheduleContext:
    """Initialize ScheduleContext."""
    n_total = tournament_config.get_n_total_events()
    roundreqs_array = np.tile(tuple(tournament_config.roundreqs.values()), (tournament_config.num_teams, 1))
    empty_schedule = np.full(n_total, -1, dtype=int)
    return ScheduleContext(
        conflict_map=event_factory.as_conflict_map(),
        event_props=event_properties,
        teams_list=np.arange(tournament_config.num_teams, dtype=int),
        teams_roundreqs_arr=roundreqs_array,
        empty_schedule=empty_schedule,
    )


@pytest.fixture
def empty_schedule(schedule_context: ScheduleContext) -> Schedule:
    """Return an empty Schedule."""
    Schedule.ctx = schedule_context
    return Schedule()


@pytest.fixture
def timeslot() -> TimeSlot:
    """Create a sample TimeSlot for testing."""
    start = parse_time_str("09:00", FMT_24H)
    stop_active = parse_time_str("09:15", FMT_24H)
    stop_cycle = parse_time_str("10:00", FMT_24H)
    TimeSlot.time_fmt = FMT_24H
    return TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
