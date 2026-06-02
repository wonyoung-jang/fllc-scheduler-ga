"""Whitelist for vulture to avoid false positives."""

from pydantic import BaseModel

from fll_scheduler_ga.constants import FitnessObjective

# ruff: noqa: B018
FitnessObjective.BREAK_TIME
FitnessObjective.LOCATION_CONSISTENCY
FitnessObjective.OPPONENT_VARIETY
BaseModel.validate  # ty:ignore[deprecated]
