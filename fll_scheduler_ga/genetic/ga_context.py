"""Context for the genetic algorithm parts."""

import logging
from dataclasses import dataclass

from ..config.config import TournamentConfig
from ..config.ga_parameters import GaParameters
from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..genetic.fitness import FitnessEvaluator
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import Selection


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    config: TournamentConfig
    ga_params: GaParameters
    event_factory: EventFactory
    team_factory: TeamFactory
    evaluator: FitnessEvaluator
    repairer: Repairer
    nsga3: NSGA3
    logger: logging.Logger
    selections: tuple[Selection]
    crossovers: tuple[Crossover]
    mutations: tuple[Mutation]
