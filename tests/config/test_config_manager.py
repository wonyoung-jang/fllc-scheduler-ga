"""Tests for config_manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fll_scheduler_ga.config.config_manager import ConfigManager

if TYPE_CHECKING:
    from pathlib import Path


def test_config_manager_initialization(tmp_path: Path) -> None:
    """Test ConfigManager initialization and default creation."""
    config_dir = tmp_path / "configs"
    default_tpl = tmp_path / "default.json"
    default_tpl.touch()

    # Init with empty directory
    cm = ConfigManager(directory=config_dir, default_template=default_tpl, active_config=tmp_path / "active.txt")
    assert (config_dir / "default.json").exists()
    assert cm.get_active_config().name == "default.json"


def test_config_manager_operations(tmp_path: Path) -> None:
    """Test add, get, list, set operations."""
    config_dir = tmp_path / "configs"
    default_tpl = tmp_path / "default.json"
    default_tpl.touch()

    cm = ConfigManager(directory=config_dir, default_template=default_tpl, active_config=tmp_path / "active.txt")

    # Add config
    new_cfg = tmp_path / "custom.json"
    new_cfg.touch()
    cm.add_config(new_cfg, "my_custom")
    assert (config_dir / "my_custom.json").exists()

    # Get by index/name
    cm.refresh_list()
    assert cm.get_config("0")
    assert cm.get_config("my_custom.json").name == "my_custom.json"

    # Set active
    cm.set_active_config("1")
    assert cm.get_active_config().name == "my_custom.json"

    # Error handling
    match_get_config_error = f"Index 99 out of range 0-{len(cm.available) - 1}"
    with pytest.raises(ValueError, match=match_get_config_error):
        cm.get_config("99")
    match_file_not_found = "Configuration 'nonexistent' not found."
    with pytest.raises(FileNotFoundError, match=match_file_not_found):
        cm.get_config("nonexistent")
