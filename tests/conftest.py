"""Fixtures for testing fll_scheduler_ga package."""

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from fll_scheduler_ga.adapter.schema import build_app_config_model
from fll_scheduler_ga.domain.model import EventProperties, EventRepository, Schedule, ScheduleContext, TimeSlot
from fll_scheduler_ga.service.config_builder import TIME_FORMAT_MAP, AppConfig, AppConfigBuilder, _parse_time_str
from fll_scheduler_ga.service.context_builder import build_evt_prop, build_evt_repo

if TYPE_CHECKING:
    from pathlib import Path

    from fll_scheduler_ga.domain.model import TournamentConfig

FMT_24H: str = TIME_FORMAT_MAP[24]
FMT_12H: str = TIME_FORMAT_MAP[12]


@pytest.fixture
def minimal_config_dict() -> dict[str, Any]:
    """Return a minimal valid configuration dictionary."""
    return {
        "genetic": {
            "parameters": {"population_size": 4, "generations": 2},
            "operator": {
                "crossover": {"types": ["KPoint"], "k_vals": [1]},
                "mutation": {"types": ["SwapTeam_CrossTimeLocation"]},
            },
            "stagnation": {"enable": False},
        },
        "runtime": {"seed_file": "test_seed.pkl"},
        "io": {"imports": {}, "exports": {}},
        "fitness": {},
        "tournament": {
            "teams": 4,
            "locations": [{"name": "Room", "count": 1, "sides": 1}, {"name": "Table", "count": 2, "sides": 2}],
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
    m = build_app_config_model(config_file)
    return AppConfigBuilder(m).build()


@pytest.fixture
def tournament_config(app_config: AppConfig) -> TournamentConfig:
    """Return the TournamentConfig."""
    return app_config.tournament


@pytest.fixture
def evt_repo(tournament_config: TournamentConfig) -> EventRepository:
    """Return an EventRepository."""
    return build_evt_repo(tournament_config.rounds)


@pytest.fixture
def evt_prop(evt_repo: EventRepository) -> EventProperties:
    """Return EventProperties."""
    return build_evt_prop(evt_repo.mapping)


@pytest.fixture
def schedule_context(
    tournament_config: TournamentConfig, evt_repo: EventRepository, evt_prop: EventProperties
) -> ScheduleContext:
    """Initialize ScheduleContext."""
    n_total = tournament_config.n_total_events
    roundreqs_array = np.tile(tuple(tournament_config.roundreqs.values()), (tournament_config.nteam, 1))
    empty_schedule = np.full(n_total, -1, dtype=int)
    return ScheduleContext(
        conflict_map=evt_repo.conflict_map,
        roundtype_idx=evt_prop.roundtype_idx,
        teams_list=np.arange(tournament_config.nteam, dtype=int),
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
    start = _parse_time_str("09:00", FMT_24H)
    stop_active = _parse_time_str("09:15", FMT_24H)
    stop_cycle = _parse_time_str("10:00", FMT_24H)
    TimeSlot.fmt = FMT_24H
    return TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
