"""Whitelist for vulture to avoid false positives."""

from fll_scheduler_ga.api.main import app

from fll_scheduler_ga.config.constants import FitnessObjective

app  # noqa: B018

FitnessObjective.BREAK_TIME  # noqa: B018
FitnessObjective.LOCATION_CONSISTENCY  # noqa: B018
FitnessObjective.OPPONENT_VARIETY  # noqa: B018
