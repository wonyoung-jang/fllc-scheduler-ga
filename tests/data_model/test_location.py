"""Unit tests for Location."""

import pytest

from fll_scheduler_ga.data_model.location import Location, LocationModelsParser


def test_location_str() -> None:
    """Test the string representation of Location."""
    loc1 = Location(idx=0, locationtype="Room", name=1, side=-1, teams_per_round=1)
    loc2 = Location(idx=1, locationtype="Table", name=1, side=1, teams_per_round=2)
    assert str(loc1) == "Room A"
    assert str(loc2) == "Table A1"


def test_location_hash() -> None:
    """Test the hashing of Location."""
    loc1 = Location(idx=0, locationtype="Room", name=2, side=-1, teams_per_round=1)
    loc2 = Location(idx=1, locationtype="Room", name=2, side=-1, teams_per_round=2)
    loc3 = Location(idx=2, locationtype="Room", name=2, side=1, teams_per_round=1)
    assert hash(loc1) == hash(loc1)
    assert hash(loc1) == hash(loc2)
    assert hash(loc1) != hash(loc3)
    assert hash(loc2) != hash(loc3)


def test_null_location() -> None:
    """Test the null location."""
    null_loc = Location()
    assert null_loc.idx == 0
    assert null_loc.locationtype == "Null"
    assert null_loc.name == 1
    assert null_loc.side == -1
    assert null_loc.teams_per_round == 1
    assert str(null_loc) == "Null A"


def test_location_models_parser_empty() -> None:
    """Test LocationModelsParser with no location models."""
    parser = LocationModelsParser(models=[])
    with pytest.raises(ValueError, match=r"No locations defined in the configuration file."):
        parser.parse()
