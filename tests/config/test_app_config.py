"""Tests for app_config."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from fll_scheduler_ga.config.app_config import AppConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_time_format_inference_error(minimal_config_dict: dict, tmp_path: Path) -> None:
    """Test error when mixed time formats are used."""
    minimal_config_dict["tournament"]["rounds"][0]["start_time"] = "09:00 AM"  # 12H
    minimal_config_dict["tournament"]["rounds"][0]["stop_time"] = "09:30"  # 24H

    config_file = tmp_path / "bad_config.json"
    with config_file.open("w") as f:
        json.dump(minimal_config_dict, f)

    with pytest.raises(ValueError, match=r"Conflicting time formats found in configuration times."):
        AppConfig.build(config_file)
