"""Whitelist for vulture to avoid false positives."""

from pydantic import BaseModel

from fll_scheduler_ga.constants import FitnessObjective

FitnessObjective.BREAK_TIME  # noqa: B018
FitnessObjective.LOCATION_CONSISTENCY  # noqa: B018
FitnessObjective.OPPONENT_VARIETY  # noqa: B018
BaseModel.validate  # noqa: B018  # ty:ignore[deprecated]
