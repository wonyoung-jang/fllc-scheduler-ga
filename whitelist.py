"""Whitelist for vulture to avoid false positives."""

from fll_scheduler_ga.config.constants import FitnessObjective
from fll_scheduler_ga.config.schemas import GaParameters, TournamentRound
from fll_scheduler_ga.data_model.event import Event, EventProperties

FitnessObjective.BREAK_TIME  # noqa: B018
FitnessObjective.LOCATION_CONSISTENCY  # noqa: B018
FitnessObjective.OPPONENT_VARIETY  # noqa: B018

GaParameters.validate  # noqa: B018
TournamentRound.validate_slots_empty  # noqa: B018

EventProperties.all_props  # noqa: B018
Event.model_config  # noqa: B018
