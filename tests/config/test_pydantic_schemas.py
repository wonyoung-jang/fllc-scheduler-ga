"""Tests for pydantic_schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fll_scheduler_ga.config.app_config import get_team_identities, get_teams_list
from fll_scheduler_ga.config.pydantic_schemas import FitnessModel, ImportModel, RoundModel, TeamsModel


def test_schemas_validation() -> None:
    """Test Pydantic model validations."""
    # ImportModel
    with pytest.raises(ValidationError):
        ImportModel(seed_pop_sort="invalid")

    with pytest.raises(ValidationError):
        ImportModel(seed_island_strategy="invalid")

    # FitnessModel
    fm = FitnessModel(weight_mean=0, weight_variation=0, weight_range=0, minbreak_penalty=-1)
    fm_fitness_tuple = fm.get_fitness_tuple()
    fm_obj_weights = fm.get_obj_weights()
    assert all(w == 1 / 3 for w in fm_fitness_tuple)
    assert all(w == 1.0 / 1.0 for w in fm_obj_weights)
    assert fm.weight_mean == 1.0  # Default
    assert fm.minbreak_penalty > 0

    # RoundModel
    with pytest.raises(ValidationError):
        RoundModel(roundtype="R1", start_time="", times=[])  # Missing times

    with pytest.raises(ValidationError):
        RoundModel(roundtype="R1", start_time="", stop_time="09:00")  # Stop without start

    with pytest.raises(ValidationError):
        RoundModel(
            roundtype="R1",
            start_time="09:00",
            stop_time="10:00",
            duration_active=10,
            duration_cycle=5,
        )  # Active > Cycle

    # TeamsModel
    tm = TeamsModel(teams=5)
    teams_list = get_teams_list(tm.teams)
    assert len(teams_list) == 5

    ids = get_team_identities(teams_list)
    assert ids[1] == "1"
