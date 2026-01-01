"""Fixtures for testing fll_scheduler_ga package."""

from typing import Any

import pytest

from fll_scheduler_ga.data_model.timeslot import TIME_FORMAT_MAP, TimeSlot, parse_time_str

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
        "io": {
            "imports": {},
            "exports": {},
        },
        "fitness": {},
        "tournament": {
            "teams": 4,
            "locations": [{"name": "Table", "count": 2, "sides": 2}],
            "rounds": [
                {
                    "roundtype": "Match",
                    "location": "Table",
                    "rounds_per_team": 1,
                    "teams_per_round": 2,
                    "start_time": "09:00",
                    "stop_time": "09:30",
                    "duration_cycle": 5,
                    "duration_active": 3,
                }
            ],
        },
    }


@pytest.fixture
def timeslot() -> TimeSlot:
    """Create a sample TimeSlot for testing."""
    start = parse_time_str("09:00", FMT_24H)
    stop_active = parse_time_str("09:15", FMT_24H)
    stop_cycle = parse_time_str("10:00", FMT_24H)
    TimeSlot.time_fmt = FMT_24H
    return TimeSlot(idx=0, start=start, stop_active=stop_active, stop_cycle=stop_cycle)
