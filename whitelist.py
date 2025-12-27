"""Whitelist for vulture to avoid false positives."""

from pydantic import BaseModel

from fll_scheduler_ga.config.constants import FitnessObjective
from fll_scheduler_ga.data_model.event import EventProperties

FitnessObjective.BREAK_TIME  # noqa: B018
FitnessObjective.LOCATION_CONSISTENCY  # noqa: B018
FitnessObjective.OPPONENT_VARIETY  # noqa: B018

BaseModel.validate  # noqa: B018  # ty:ignore[deprecated]
EventProperties.all_props  # noqa: B018
