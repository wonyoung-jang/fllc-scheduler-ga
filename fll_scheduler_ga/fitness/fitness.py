"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..config.constants import EPSILON, FITNESS_PENALTY, FitnessObjective

if TYPE_CHECKING:
    from ..config.schemas import TournamentConfig
    from ..data_model.event import EventProperties
    from ..data_model.schedule import Schedule
    from .benchmark import FitnessBenchmark

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

    objectives: list[FitnessObjective] = None
    penalty: float = FITNESS_PENALTY

    max_events_per_team: int = 0

    max_int: ClassVar[int] = np.iinfo(np.int64).max
    min_int: ClassVar[int] = -1
    n_teams: ClassVar[int]
    n_objs: ClassVar[int]
    match_roundtypes: ClassVar[np.ndarray]
    min_matches: ClassVar[int]

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives = list(FitnessObjective)
        self.max_events_per_team = self.config.max_events_per_team
        FitnessEvaluator.n_teams = self.config.num_teams
        FitnessEvaluator.n_objs = len(self.objectives)
        FitnessEvaluator.match_roundtypes = np.array(
            [rt_idx for rt_idx, tpr in self.config.round_idx_to_tpr.items() if tpr == 2]
        )
        FitnessEvaluator.min_matches = min(r.rounds_per_team for r in self.config.rounds if r.teams_per_round == 2)

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
        roundtype_ids = self.event_properties.roundtype_idx[lookup_events]
        # Invalidate non-existent events
        starts[~valid_events] = FitnessEvaluator.max_int
        stops[~valid_events] = FitnessEvaluator.max_int
        loc_ids[~valid_events] = FitnessEvaluator.min_int
        paired_evt_ids[~valid_events] = FitnessEvaluator.min_int
        roundtype_ids[~valid_events] = FitnessEvaluator.min_int
        # Calculate scores for each objective
        team_fitnesses[:, :, 0] = self.score_break_time(starts, stops)
        team_fitnesses[:, :, 1] = self.score_loc_consistency(loc_ids, roundtype_ids)
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
        breaks_minutes = breaks_seconds / 60
        breaks_minutes[~valid_consecutive] = np.nan
        # Identify overlaps
        overlap_mask = np.any(breaks_minutes < 0, axis=2)

        mean_break = np.nanmean(breaks_minutes, axis=2)
        mean_break = np.nan_to_num(mean_break, nan=EPSILON)
        mean_zero_mask = mean_break == 0

        std_dev = np.nanstd(breaks_minutes, axis=2)
        std_dev = np.nan_to_num(std_dev, nan=EPSILON)

        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)

        zeros = np.sum(breaks_minutes == 0, axis=2)
        zeros_penalty = self.penalty**zeros

        final_scores = ratio * zeros_penalty

        final_scores[mean_zero_mask] = 0
        final_scores[overlap_mask] = 0

        return final_scores / (self.benchmark.best_timeslot_score or 1.0)

    def score_loc_consistency(self, loc_ids: np.ndarray, roundtype_ids: np.ndarray) -> np.ndarray:
        """Calculate location consistency score."""
        n_pop, n_teams, n_rounds = loc_ids.shape

        # Create mask for valid locations in match roundtypes
        match_roundtypes = FitnessEvaluator.match_roundtypes
        n_rt = match_roundtypes.size
        if n_rt == 0:
            return np.ones((n_pop, n_teams), dtype=float)
        match_rt_mask = np.isin(roundtype_ids, match_roundtypes) & (loc_ids >= 0)

        # Total matches per team
        total_matches_per_team = match_rt_mask.sum(axis=2)

        # Get all unique locations per team
        max_loc = loc_ids.max() + 1
        loc_one_hot = np.full((n_pop, n_teams, n_rounds, max_loc), fill_value=False, dtype=bool)

        # Vectorized assignment using advanced indexing
        pop_idx, team_idx, round_idx = np.where(match_rt_mask)
        loc_vals = loc_ids[match_rt_mask]
        loc_one_hot[pop_idx, team_idx, round_idx, loc_vals] = True

        # Union: any location used across all rounds
        union_mask = loc_one_hot.any(axis=2)  # (n_pop, n_teams, max_loc)
        total_unique_locs = union_mask.sum(axis=2)  # (n_pop, n_teams)

        # Primary Score: fewer unique locations is better
        max_negative = total_matches_per_team - 1
        negative = total_unique_locs - 1
        primary_scores = np.where(max_negative <= 0, 1.0, 1.0 - (negative / np.maximum(max_negative, 1)))

        # Bonus Score: intersection across round types
        # Create masks
        # Expand roundtype_ids to match match_roundtypes for broadcasting
        # (n_rt, n_pop, n_teams, n_rounds)
        rt_comparison = roundtype_ids[None, :, :, :] == match_roundtypes[:, None, None, None]
        rt_valid = rt_comparison & (loc_ids[None, :, :, :] >= 0)

        # Get indices where each roundtype has valid locations
        rt_idx_all, pop_idx_all, team_idx_all, round_idx_all = np.where(rt_valid)
        loc_vals_all = np.broadcast_to(loc_ids[None, :, :, :], (n_rt, n_pop, n_teams, n_rounds))[rt_valid]

        # Create a 5D array: (n_roundtypes, n_pop, n_teams, n_rounds, max_loc), then assign
        rt_masks_5d = np.zeros((n_rt, n_pop, n_teams, n_rounds, max_loc), dtype=bool)
        rt_masks_5d[rt_idx_all, pop_idx_all, team_idx_all, round_idx_all, loc_vals_all] = True

        # Aggregate (n_rt, n_pop, n_teams, max_loc), take any across rounds
        rt_loc_masks = rt_masks_5d.any(axis=3)

        # Intersection: locations present in all round types
        intersection_mask = rt_loc_masks.all(axis=0) if n_rt > 1 else rt_loc_masks[0]  # (n_pop, n_teams, max_loc)
        overlap_count = intersection_mask.sum(axis=2)  # (n_pop, n_teams)

        min_matches = FitnessEvaluator.min_matches
        bonus_scores = np.where(min_matches > 0, overlap_count / min_matches, 1.0)

        # Combine with weights
        wp = 0.9
        wb = 0.1
        norm_max = (1.0 * wp) + (0.5 * wb)
        return ((primary_scores * wp) + (bonus_scores * wb)) / norm_max

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
