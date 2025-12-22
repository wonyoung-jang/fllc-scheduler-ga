"""Base class for fitness evaluators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.config.constants import EPSILON, FitnessObjective

if TYPE_CHECKING:
    from fll_scheduler_ga.config.schemas import FitnessModel, TournamentConfig
    from fll_scheduler_ga.data_model.event import EventProperties
    from fll_scheduler_ga.fitness.benchmark import FitnessBenchmark

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessBase(ABC):
    """Base class for fitness evaluators."""

    # Configurations
    config: TournamentConfig
    event_properties: EventProperties
    benchmark: FitnessBenchmark
    model: FitnessModel
    # Globals
    max_int: int = np.iinfo(np.int64).max
    n_objectives: int = len(tuple(FitnessObjective))
    epsilon: float = EPSILON
    # TournamentConfig
    n_teams: int = 0
    n_max_events: int = 0
    n_match_rt: int = 0
    n_single_rt: int = 0
    single_roundtypes: np.ndarray = field(default_factory=lambda: np.array([]))
    match_roundtypes: np.ndarray = field(default_factory=lambda: np.array([]))
    rt_array: np.ndarray = field(default_factory=lambda: np.array([]))
    # EventProperties
    _start: np.ndarray = field(default_factory=lambda: np.array([]))
    _stop_active: np.ndarray = field(default_factory=lambda: np.array([]))
    _stop_cycle: np.ndarray = field(default_factory=lambda: np.array([]))
    _loc_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    _paired_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    _roundtype_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    # FitnessBenchmark
    benchmark_oppoenents: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark_best_timeslot_score: float = 0.0
    # FitnessModel
    loc_weight_rounds_inter: float = 0.0
    loc_weight_rounds_intra: float = 0.0
    agg_weights: tuple[float, ...] = ()
    obj_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    min_fitness_weight: float = 0.0
    minbreak_target: int = 0
    minbreak_penalty: float = 0.0
    zeros_penalty: float = 0.0

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        # Initialize from TournamentConfig
        self.n_teams = self.config.num_teams
        self.n_max_events = self.config.max_events_per_team
        rti_to_tpr = self.config.round_idx_to_tpr
        self.single_roundtypes = np.array([rti for rti, tpr in rti_to_tpr.items() if tpr == 1])
        self.match_roundtypes = np.array([rti for rti, tpr in rti_to_tpr.items() if tpr == 2])
        self.n_single_rt = self.single_roundtypes.size
        self.n_match_rt = self.match_roundtypes.size
        max_rt_idx = self.match_roundtypes.max() if self.match_roundtypes.size > 0 else -1
        self.rt_array = np.full(max_rt_idx + 1, -1, dtype=int)
        for i, rt in enumerate(self.match_roundtypes):
            self.rt_array[rt] = i
        # Initialize from EventProperties
        _ep = self.event_properties
        self._start = _ep.start
        self._stop_active = _ep.stop_active
        self._stop_cycle = _ep.stop_cycle
        self._loc_idx = _ep.loc_idx
        self._paired_idx = _ep.paired_idx
        self._roundtype_idx = _ep.roundtype_idx
        # Initialize from FitnessBenchmark
        self.benchmark_oppoenents = self.benchmark.opponents
        self.benchmark_best_timeslot_score = self.benchmark.best_timeslot_score
        # Initialize from FitnessModel
        self.loc_weight_rounds_inter = self.model.loc_weight_rounds_inter
        self.loc_weight_rounds_intra = self.model.loc_weight_rounds_intra
        self.agg_weights = self.model.get_fitness_tuple()
        self.obj_weights = np.array(self.model.get_obj_weights(), dtype=float)
        self.min_fitness_weight = self.model.min_fitness_weight
        self.minbreak_target = self.model.minbreak_target
        self.minbreak_penalty = self.model.minbreak_penalty
        self.zeros_penalty = self.model.zeros_penalty

    @abstractmethod
    def evaluate(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the fitness of schedule(s).

        Args:
            arr: Array of schedule representation(s)

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (schedule fitness(es), team fitnesses)

        """

    @abstractmethod
    def get_team_events(self, arr: np.ndarray) -> np.ndarray:
        """Get team events from schedule representation(s).

        Args:
            arr: Array of schedule representation(s)

        Returns:
            np.ndarray: Array of team events

        """

    def _slice_event_properties(self, team_events: np.ndarray) -> tuple[np.ndarray, ...]:
        """Slice event properties arrays based on team events.

        Args:
            team_events: Array of event IDs for each team

        Returns:
            tuple[np.ndarray, ...]: Tuple of sliced event properties arrays

        """
        return (
            self._start[team_events],
            self._stop_active[team_events],
            self._stop_cycle[team_events],
            self._loc_idx[team_events],
            self._paired_idx[team_events],
            self._roundtype_idx[team_events],
        )

    def _aggregate_team_scores(self, team_fitnesses: np.ndarray, team_axis: int) -> np.ndarray:
        """Aggregate team fitness scores into schedule fitness scores.

        Args:
            team_fitnesses: Array of team fitness scores
            team_axis: Axis along which teams are indexed

        Returns:
            np.ndarray: Aggregated schedule fitness scores

        """
        min_s = team_fitnesses.min(axis=team_axis)
        mean_s = team_fitnesses.mean(axis=team_axis)
        min_fitness_weight = self.min_fitness_weight
        mean_s = (mean_s * (1.0 - min_fitness_weight)) + (min_s * min_fitness_weight)
        mean_s[mean_s == 0] = self.epsilon

        stddev_s = team_fitnesses.std(axis=team_axis)
        coeff_s = stddev_s / mean_s
        vari_s = 1.0 / (1.0 + coeff_s)

        max_for_ptp = team_fitnesses.max(axis=team_axis)
        min_for_ptp = team_fitnesses.min(axis=team_axis)
        ptp = max_for_ptp - min_for_ptp
        range_s = 1.0 / (1.0 + ptp)

        mw, vw, rw = self.agg_weights
        schedule_fitnesses = (mean_s * mw) + (vari_s * vw) + (range_s * rw)

        return schedule_fitnesses * self.obj_weights
