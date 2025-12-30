"""Tests for pydantic_schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fll_scheduler_ga.config.pydantic_schemas import FitnessModel, ImportModel, RoundModel, TeamsModel


def test_schemas_validation() -> None:
    """Test Pydantic model validations."""
    # ImportModel
    with pytest.raises(ValidationError):
        ImportModel(seed_pop_sort="invalid")

    # FitnessModel
    fm = FitnessModel(weight_mean=0, weight_variation=0, weight_range=0, minbreak_penalty=-1)
    assert fm.weight_mean == 1.0  # Default
    assert fm.minbreak_penalty > 0

    # RoundModel
    with pytest.raises(ValidationError):
        RoundModel(roundtype="R1", start_time="", times=[])  # Missing times

    with pytest.raises(ValidationError):
        RoundModel(
            roundtype="R1", start_time="09:00", stop_time="10:00", duration_active=10, duration_cycle=5
        )  # Active > Cycle

    # TeamsModel
    tm = TeamsModel(teams=5)
    assert len(tm) == 5

    ids = tm.get_team_ids()
    assert ids[1] == "1"
