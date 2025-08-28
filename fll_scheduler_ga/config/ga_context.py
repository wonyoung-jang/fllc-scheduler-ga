"""Context for the genetic algorithm parts."""

from dataclasses import dataclass

from ..data_model.event import EventFactory
from ..data_model.team import TeamFactory
from ..genetic.fitness import FitnessEvaluator
from ..operators.crossover import Crossover
from ..operators.mutation import Mutation
from ..operators.nsga3 import NSGA3
from ..operators.repairer import Repairer
from ..operators.selection import Selection
from .app_config import AppConfig


@dataclass(slots=True)
class GaContext:
    """Hold static context for the genetic algorithm."""

    app_config: AppConfig
    event_factory: EventFactory
    team_factory: TeamFactory
    evaluator: FitnessEvaluator
    repairer: Repairer
    nsga3: NSGA3
    selections: tuple[Selection, ...]
    crossovers: tuple[Crossover, ...]
    mutations: tuple[Mutation, ...]
