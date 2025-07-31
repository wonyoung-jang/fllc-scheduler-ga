"""Context for the genetic algorithm parts."""

import logging
from dataclasses import dataclass, field

from ..config.app_config import AppConfig
from ..config.config import TournamentConfig
from ..config.ga_parameters import GaParameters
from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..genetic.fitness import FitnessEvaluator
from ..operators.crossover import Crossover, build_crossovers
from ..operators.mutation import Mutation, build_mutations
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import Selection, build_selections


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    team_factory: TeamFactory
    evaluator: FitnessEvaluator
    repairer: Repairer
    logger: logging.Logger
    nsga3: NSGA3
    config: TournamentConfig = field(init=False)
    ga_params: GaParameters = field(init=False)
    selections: tuple[Selection] = field(init=False)
    crossovers: tuple[Crossover] = field(init=False)
    mutations: tuple[Mutation] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the context."""
        self.config = self.app_config.tournament
        self.ga_params = self.app_config.ga_params
        self.selections = tuple(build_selections(self.app_config))
        self.crossovers = tuple(build_crossovers(self.app_config, self.team_factory, self.event_factory))
        self.mutations = tuple(build_mutations(self.app_config))
