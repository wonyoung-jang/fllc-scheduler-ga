"""Seed data I/O for genetic algorithm."""

import pickle
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


logger = getLogger(__name__)


def save_pkl(path: Path, data: Any) -> None:
    """Save arbitrary data to a pickle file."""
    try:
        logger.debug("Saving %s to: %s", data.__class__.__qualname__, path)
        with path.open("wb") as f:
            pickle.dump(data, f)
    except OSError, pickle.PicklingError, EOFError:
        logger.exception("Error saving %s to seed file: %s", path, data.__class__.__qualname__)


def load_pkl(path: Path) -> Any | None:
    """Load arbitrary data from a pickle file."""
    try:
        logger.debug("Loading data from: %s", path)
        with path.open("rb") as f:
            return pickle.load(f)
    except OSError, pickle.UnpicklingError, ModuleNotFoundError, EOFError:
        logger.warning("Could not load or parse pickle file at: %s", path)
        return None
