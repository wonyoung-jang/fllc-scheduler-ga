"""Fitness evaluator for a single FLL schedule."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

import numpy as np

from ..config.constants import EPSILON
from .fitness_base import FitnessBase

logger = getLogger(__name__)


@dataclass(slots=True)
class FitnessEvaluatorSingle(FitnessBase):
    """Calculates the fitness of a single schedule."""

    def evaluate(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a single schedule.

        Args:
            arr (np.ndarray): Shape (num_events,). The core data.

        Returns:
            np.ndarray: Final fitness scores for the schedule. Shape (num_objectives,).
            np.ndarray: All team scores for the schedule. Shape (num_teams, num_objectives).

        """
        # Get team-events mapping
        team_events = self.get_team_events(arr)

        # Slice event properties
        starts, stops_active, stops_cycle, loc_ids, paired_evt_ids, roundtype_ids = self._slice_event_properties(
            team_events=team_events
        )

        # Preallocate arrays
        team_fits_shape = (self.n_teams, self.n_objectives)
        team_fitnesses = np.zeros(team_fits_shape, dtype=float)

        # Calculate scores for each objective
        team_fitnesses[:, 0] = self.score_break_time(starts, stops_active, stops_cycle)
        team_fitnesses[:, 1] = self.score_loc_consistency(loc_ids, roundtype_ids)
        team_fitnesses[:, 2] = self.score_opp_variety(paired_evt_ids, arr)

        # Aggregate team scores into schedule scores
        schedule_fitnesses = self._aggregate_team_scores(team_fitnesses, team_axis=0)

        return schedule_fitnesses, team_fitnesses

    def get_team_events(self, arr: np.ndarray) -> np.ndarray:
        """Invert the (event -> team) mapping to a (team -> events) mapping."""
        # Preallocate the team-events array
        team_events = np.full((self.n_teams, self.config.max_events_per_team), -1, dtype=int)

        # Get indices of scheduled events
        event_indices = np.nonzero(arr >= 0)[0]

        # Handle the case with no scheduled events
        if event_indices.size == 0:
            team_events.fill(0)
            return team_events

        # Get team assignments for scheduled events
        team_indices = arr[event_indices]

        # Sort by team index to group events by team
        order = np.argsort(team_indices)
        sorted_team_indices = team_indices[order]
        sorted_event_indices = event_indices[order]

        # Count events per team
        counts = np.bincount(sorted_team_indices, minlength=self.n_teams)

        # Compute group starts for each team
        group_starts = np.zeros_like(counts, dtype=int)
        group_starts[1:] = np.cumsum(counts[:-1])

        # Compute within-group indices
        repeated_starts = group_starts.repeat(counts)
        within_group_indices = np.arange(sorted_event_indices.size, dtype=int) - repeated_starts

        # Filter to only valid slots within max_events_per_team
        valid_mask = within_group_indices < self.config.max_events_per_team
        team_idx_final = sorted_team_indices[valid_mask]
        slot_idx_final = within_group_indices[valid_mask]
        event_idx_final = sorted_event_indices[valid_mask]
        team_events[team_idx_final, slot_idx_final] = event_idx_final

        return team_events

    def score_break_time(self, starts: np.ndarray, stops_active: np.ndarray, stops_cycle: np.ndarray) -> np.ndarray:
        """Vectorized break time scoring."""
        # Sort events by start time
        order = np.argsort(starts, axis=1)
        starts_sorted = np.take_along_axis(starts, order, axis=1)
        stops_active_sorted = np.take_along_axis(stops_active, order, axis=1)
        stops_cycle_sorted = np.take_along_axis(stops_cycle, order, axis=1)

        # Calculate breaks between consecutive events
        start_next = starts_sorted[:, 1:]
        stop_active_curr = stops_active_sorted[:, :-1]
        stop_cycle_curr = stops_cycle_sorted[:, :-1]

        # Calculate break durations in minutes
        breaks_active_seconds = np.subtract(start_next, stop_active_curr)
        breaks_active_minutes = breaks_active_seconds / 60

        breaks_cycle_seconds = np.subtract(start_next, stop_cycle_curr)
        breaks_cycle_minutes = breaks_cycle_seconds / 60

        # Identify overlaps
        overlap_mask = (breaks_cycle_minutes < 0).any(axis=1)

        # Calculate mean
        valid_mask = breaks_cycle_minutes >= 0
        count = valid_mask.sum(axis=1, dtype=int)

        mean_break = breaks_cycle_minutes.sum(axis=1) / count
        mean_break_zero_mask = mean_break == 0
        mean_break[mean_break_zero_mask] = EPSILON

        # Calculate standard deviation
        diff_sq: np.ndarray = np.square(breaks_cycle_minutes - mean_break[:, np.newaxis])
        variance = diff_sq.sum(axis=1) / count
        std_dev: np.ndarray = np.sqrt(variance)

        # Calculate coefficient of variation
        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)

        # Apply minimum break penalty
        minbreak_count = (breaks_active_minutes < self.model.minbreak_target).sum(axis=1)
        where_breaks_lt_target = (breaks_active_minutes < self.model.minbreak_target) & (breaks_active_minutes > 0)
        max_diff_breaktimes = np.zeros_like(minbreak_count)
        if where_breaks_lt_target.any():
            diffs = self.model.minbreak_target - breaks_active_minutes
            diffs[~where_breaks_lt_target] = 0.0
            max_diff_breaktimes = diffs.max(axis=1) / self.model.minbreak_target
        minbreak_exp = minbreak_count + max_diff_breaktimes
        minbreak_penalty = self.model.minbreak_penalty**minbreak_exp

        # Apply penalties for zero breaks
        zeros_count = (breaks_cycle_minutes == 0).sum(axis=1)
        zeros_penalty = self.model.zeros_penalty**zeros_count

        # Apply penalties
        final_scores = ratio * zeros_penalty * minbreak_penalty
        final_scores[mean_break_zero_mask] = 0.0
        final_scores[overlap_mask] = 0.0

        return final_scores / self.benchmark.best_timeslot_score

    def score_loc_consistency(self, loc_ids: np.ndarray, roundtype_ids: np.ndarray) -> np.ndarray:
        """Calculate location consistency score, prioritizing inter-round over intra-round consistency."""
        n_teams, _ = loc_ids.shape
        match_roundtypes = self.match_roundtypes
        shape = n_teams

        # Consistency score is only meaningful with 1+ match round types
        if self.n_match_rt < 1:
            return np.ones(shape, dtype=float)

        # Create a (team, rt, loc) boolean mask
        max_loc_idx = loc_ids.max()
        # No locations scheduled
        if max_loc_idx < 0:
            return np.ones(shape, dtype=float)

        max_rt_id = max(roundtype_ids.max(), match_roundtypes.max())
        is_match_rt_lookup = np.zeros(max_rt_id + 1, dtype=bool)
        is_match_rt_lookup[match_roundtypes] = True
        match_rt_mask = is_match_rt_lookup[roundtype_ids] & (loc_ids >= 0)

        team_indices, _ = match_rt_mask.nonzero()
        loc_vals = loc_ids[match_rt_mask]
        rt_values = roundtype_ids[match_rt_mask]
        mapped_rt_indices = self.rt_array[rt_values]

        # Inter-Round Consistency
        inter_round_scores = np.ones(shape, dtype=float)

        # Build count array: (team, rt, loc)
        rt_loc_counts = np.zeros((n_teams, self.n_match_rt, max_loc_idx + 1), dtype=int)
        rt_loc_counts[team_indices, mapped_rt_indices, loc_vals] = 1

        # A team participated in a round type if its location counts for that RT are > 0.
        participated_in_rt_counts = rt_loc_counts.sum(axis=2, dtype=int)
        participated_in_rt = participated_in_rt_counts > 0

        # A location is in the intersection if its count across RTs equals the number of participated RTs.
        num_participated_rts = participated_in_rt.sum(axis=1, dtype=float)

        if self.n_match_rt >= 2:
            # Create a boolean mask of used locations (count > 0)
            loc_used_in_rt_mask = rt_loc_counts > 0

            # The result is the number of different round types a location was used in.
            loc_usage_across_rts = loc_used_in_rt_mask.sum(axis=1, dtype=float)

            # A location is in intersection if used in number of RTs equal to total number of RTs team participated in.
            intersection_mask = loc_usage_across_rts == num_participated_rts[:, np.newaxis]
            intersection_size = intersection_mask.sum(axis=1)

            # The union is the count of locations used in at least one round type.
            union_mask = loc_usage_across_rts > 0
            union_size = union_mask.sum(axis=1)

            # Handle the zero-division case explicitly.
            valid_union = union_size > 0
            inter_round_scores[valid_union] = intersection_size[valid_union] / union_size[valid_union]

        # Intra-Round Consistency
        unique_locs_per_rt: np.ndarray = (rt_loc_counts > 0).sum(axis=2, dtype=float)
        unique_locs_per_rt[unique_locs_per_rt == 0] = EPSILON

        scores_per_rt = 1.0 / unique_locs_per_rt
        scores_per_rt[~participated_in_rt] = 1.0

        # Handle the zero-division case explicitly.
        valid_num_rts = num_participated_rts > 0
        intra_round_scores = np.ones(shape, dtype=float)
        intra_round_scores[valid_num_rts] = (
            scores_per_rt.sum(axis=1)[valid_num_rts] / num_participated_rts[valid_num_rts]
        )

        # Final Combination
        total_matches_per_team = match_rt_mask.sum(axis=1)
        final_scores = (inter_round_scores * self.loc_weight_rounds_inter) + (
            intra_round_scores * self.loc_weight_rounds_intra
        )
        final_scores[total_matches_per_team <= 1] = 1.0

        return final_scores

    def score_opp_variety(self, paired_evt_ids: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        # Create a mask for valid opponent IDs
        valid_opp = paired_evt_ids >= 0

        # Invalidate opponent IDs for invalid events
        paired_evt_ids[~valid_opp] = 0
        invalid_opp = ~valid_opp

        # Get opponents for each team
        opponents = arr[paired_evt_ids]
        opponents[invalid_opp] = self.max_int
        opponents.sort(axis=1)

        # Changes between consecutive opponents
        prev_opponents = opponents[:, :-1]
        valid_mask = prev_opponents >= 0
        diffs = np.diff(opponents, axis=1)
        changes = diffs != 0

        # Check if there are single round types
        unique_counts = (changes & valid_mask).sum(axis=1)
        unique_counts = unique_counts + 1 if self.n_single_rt == 0 else unique_counts

        return self.benchmark.opponents[unique_counts]
