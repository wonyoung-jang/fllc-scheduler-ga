"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..config.constants import EPSILON, FITNESS_PENALTY, FitnessObjective

if TYPE_CHECKING:
    from ..config.benchmark import FitnessBenchmark
    from ..data_model.config import TournamentConfig
    from ..data_model.event import EventProperties
    from ..data_model.schedule import Schedule

logger = getLogger(__name__)

# import sys
# np.set_printoptions(threshold=sys.maxsize, linewidth=200, edgeitems=30)


@dataclass(slots=True)
class HardConstraintChecker:
    """Validates hard constraints for a schedule."""

    config: TournamentConfig

    def check(self, schedule: Schedule) -> bool:
        """Check the hard constraints of a schedule."""
        if not schedule:
            return False

        if len(schedule) != self.config.total_slots_required:
            return False

        return not schedule.any_rounds_needed()


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    benchmark: FitnessBenchmark
    event_properties: EventProperties
    max_events_per_team: int
    objectives: list[FitnessObjective] = None

    max_int: ClassVar[int] = np.iinfo(np.int64).max
    min_int: ClassVar[int] = -1
    n_teams: ClassVar[int] = None
    n_objs: ClassVar[int] = None

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives = list(FitnessObjective)
        FitnessEvaluator.n_teams = self.config.num_teams
        FitnessEvaluator.n_objs = len(self.objectives)

    def evaluate_population(self, pop_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an entire population of schedules.

        Args:
            pop_array (np.ndarray): Shape (pop_size, num_events). The core data.

        Returns:
            np.ndarray: Final fitness scores for the population. Shape (pop_size, num_objectives).
            np.ndarray: All team scores for the population. Shape (pop_size, num_teams, num_objectives).

        """
        # Dims for reference
        n_pop = pop_array.shape[0]
        # Preallocate arrays
        team_fitnesses = np.zeros((n_pop, FitnessEvaluator.n_teams, FitnessEvaluator.n_objs), dtype=float)
        # Get team-events mapping for the entire population
        valid_events, lookup_events = self.get_team_events(pop_array)
        # Slice event properties
        starts = self.event_properties.start[lookup_events]
        stops = self.event_properties.stop[lookup_events]
        loc_ids = self.event_properties.loc_idx[lookup_events]
        paired_evt_ids = self.event_properties.paired_idx[lookup_events]
        # Invalidate non-existent events
        starts[~valid_events] = FitnessEvaluator.max_int
        stops[~valid_events] = FitnessEvaluator.max_int
        loc_ids[~valid_events] = FitnessEvaluator.min_int
        paired_evt_ids[~valid_events] = FitnessEvaluator.min_int
        # Calculate scores for each objective
        team_fitnesses[:, :, 0] = self.score_break_time(starts, stops)
        team_fitnesses[:, :, 1] = self.score_loc_consistency(loc_ids)
        team_fitnesses[:, :, 2] = self.score_opp_variety(paired_evt_ids, pop_array)
        # Aggregate team scores into schedule scores
        mean_s = team_fitnesses.mean(axis=1)
        std_devs = team_fitnesses.std(axis=1)
        coeffs_of_variation = std_devs / (mean_s + EPSILON)
        vari_s = 1.0 / (1.0 + coeffs_of_variation)
        ranges = np.ptp(team_fitnesses, axis=1)
        rnge_s = 1.0 / (1.0 + ranges)
        mw, vw, rw = self.config.weights
        schedule_fitnesses = (mean_s * mw) + (vari_s * vw) + (rnge_s * rw)
        return schedule_fitnesses, team_fitnesses

    def get_team_events(self, pop_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Invert the (event -> team) mapping to a (team -> events) mapping for the entire population."""
        n_pop = pop_array.shape[0]
        team_events_pop = np.full((n_pop, FitnessEvaluator.n_teams, self.max_events_per_team), -1, dtype=int)
        sched_indices, event_indices = np.where(pop_array >= 0)
        team_indices = pop_array[sched_indices, event_indices]
        if sched_indices.size > 0:
            keys = (sched_indices * FitnessEvaluator.n_teams) + team_indices
            order = np.lexsort((event_indices, keys))
            keys_sorted = keys[order]
            group_starts = np.r_[0, np.nonzero(keys_sorted[1:] != keys_sorted[:-1])[0] + 1]
            group_lengths = np.diff(np.r_[group_starts, keys_sorted.size])
            within_sorted = np.arange(keys_sorted.size, dtype=int) - np.repeat(group_starts, group_lengths)
            within = np.empty_like(within_sorted)
            within[order] = within_sorted
            valid_mask = within < self.max_events_per_team
            if np.any(valid_mask):
                team_events_pop[
                    sched_indices[valid_mask],
                    team_indices[valid_mask],
                    within[valid_mask],
                ] = event_indices[valid_mask]
        valid_events_mask = team_events_pop >= 0
        lookup_events = np.where(valid_events_mask, team_events_pop, -1)
        return valid_events_mask, lookup_events

    def score_break_time(self, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
        """Vectorized break time scoring."""
        # Sort events by start time
        order = np.argsort(starts, axis=2)
        starts_sorted = np.take_along_axis(starts, order, axis=2)
        stops_sorted = np.take_along_axis(stops, order, axis=2)
        # Calculate breaks between consecutive events
        start_next = starts_sorted[:, :, 1:]
        stop_curr = stops_sorted[:, :, :-1]
        # Valid consecutive events must have valid start and stop times
        valid_consecutive = (start_next < FitnessEvaluator.max_int) & (stop_curr < FitnessEvaluator.max_int)
        breaks_seconds = start_next - stop_curr
        breaks_minutes = breaks_seconds / 60.0
        breaks_minutes[~valid_consecutive] = np.nan
        # Identify overlaps
        overlap_mask = np.any(breaks_minutes < 0, axis=2)

        mean_break = np.nanmean(breaks_minutes, axis=2)
        mean_break = np.nan_to_num(mean_break, nan=0.0)

        std_dev = np.nanstd(breaks_minutes, axis=2)
        std_dev = np.nan_to_num(std_dev, nan=0.0)

        coefficient = std_dev / (mean_break + EPSILON)
        ratio = 1.0 / (1.0 + coefficient)

        zeros = np.sum(breaks_minutes == 0, axis=2)
        zeros_penalty = FITNESS_PENALTY**zeros

        final_scores = ratio * zeros_penalty
        final_scores[overlap_mask] = -1.0
        final_scores /= self.benchmark.best_timeslot_score
        return final_scores

    def score_loc_consistency(self, loc_ids: np.ndarray) -> np.ndarray:
        """Vectorized location consistency scoring."""
        loc_sorted = np.sort(loc_ids, axis=2)

        valid_mask = loc_sorted[:, :, :-1] >= 0
        changes = np.diff(loc_sorted, axis=2) != 0
        meaningful_changes = np.sum(changes & valid_mask, axis=2)

        has_at_least_one = loc_sorted[:, :, -1] >= 0
        unique_counts = np.where(has_at_least_one, meaningful_changes, 0)

        return self.benchmark.locations[unique_counts]

    def score_opp_variety(self, paired_evt_ids: np.ndarray, pop_array: np.ndarray) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        n_pop = pop_array.shape[0]
        valid_opp = paired_evt_ids >= 0
        lookup_opp_events = np.where(valid_opp, paired_evt_ids, 0)

        schedule_indices = np.arange(n_pop)[:, None, None]
        opponents = pop_array[schedule_indices, lookup_opp_events]
        opponents[~valid_opp] = FitnessEvaluator.max_int

        opponents_sorted = np.sort(opponents, axis=2)

        valid_mask = opponents_sorted[:, :, :-1] >= 0
        changes = np.diff(opponents_sorted, axis=2) != 0
        meaningful_changes = np.sum(changes & valid_mask, axis=2)

        has_any = opponents_sorted[:, :, 0] >= 0
        unique_counts = np.where(has_any, meaningful_changes, 0)

        return self.benchmark.opponents[unique_counts]
