"""Tests for pydantic_schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fll_scheduler_ga.config.app_config import get_team_identities, get_teams_list
from fll_scheduler_ga.config.pydantic_schemas import (
    AggregationWeightsModel,
    FitnessModel,
    ImportModel,
    LocationWeightsModel,
    RoundModel,
)


def test_schemas_validation() -> None:
    """Test Pydantic model validations."""
    # ImportModel
    with pytest.raises(ValidationError):
        ImportModel(seed_pop_sort="invalid")

    with pytest.raises(ValidationError):
        ImportModel(seed_island_strategy="invalid")

    # FitnessModel
    fm = FitnessModel()
    fm_obj_weights = fm.objectives.get_weights_tuple()
    assert all(w == 1.0 / 1.0 for w in fm_obj_weights)

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

    # Teams
    teams = 5
    teams_list = get_teams_list(teams)
    assert len(teams_list) == 5

    ids = get_team_identities(teams_list)
    assert ids[1] == "1"


def test_fitness_aggregation_model() -> None:
    """Test AggregationWeightsModel behavior."""
    agg_weights = AggregationWeightsModel(
        mean=3,
        variation=1,
        range=1,
        min_fit=0.3,
    )
    agg_weight_tuple = agg_weights.get_weights_tuple()
    assert agg_weight_tuple == (3 / 5, 1 / 5, 1 / 5)


def test_location_weights_model() -> None:
    """Test LocationWeightsModel behavior."""
    loc_weights = LocationWeightsModel(
        inter_rounds=2.0,
        intra_rounds=1.0,
    )
    loc_weight_tuple = loc_weights.get_weights_tuple()
    assert loc_weight_tuple == (2.0 / 3.0, 1.0 / 3.0)


def test_location_weights_model_zero_sum(caplog: pytest.LogCaptureFixture) -> None:
    """Test LocationWeightsModel behavior when weights sum to zero."""
    loc_weights = LocationWeightsModel(
        inter_rounds=0.0,
        intra_rounds=0.0,
    )
    loc_weight_tuple = loc_weights.get_weights_tuple()
    assert loc_weight_tuple == (0.5, 0.5)
    assert "Location weights sum to zero: resetting to equal weights of 0.5 each." in caplog.text
