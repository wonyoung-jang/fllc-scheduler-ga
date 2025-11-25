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
    """Calculates the fitness of a population of schedules."""

    config: TournamentConfig
    event_properties: EventProperties
    benchmark: FitnessBenchmark

    objectives: ClassVar[list[FitnessObjective]] = list(FitnessObjective)
    max_events_per_team: ClassVar[int] = 0
    _penalty: ClassVar[float] = FITNESS_PENALTY
    _epsilon: ClassVar[float] = EPSILON
    max_int: ClassVar[int] = np.iinfo(np.int64).max
    min_int: ClassVar[int] = -1
    n_teams: ClassVar[int] = 0
    n_objs: ClassVar[int] = 0
    match_roundtypes: ClassVar[np.ndarray] = None
    rt_array: ClassVar[np.ndarray] = None
    loc_weight_rounds_inter: ClassVar[float] = 0.9
    loc_weight_rounds_intra: ClassVar[float] = 0.1

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        FitnessEvaluator.objectives = list(FitnessObjective)
        FitnessEvaluator.max_events_per_team = self.config.max_events_per_team
        FitnessEvaluator.n_teams = self.config.num_teams
        FitnessEvaluator.n_objs = len(self.objectives)
        match_rts = np.array([rt_idx for rt_idx, tpr in self.config.round_idx_to_tpr.items() if tpr == 2])
        max_rt_idx = match_rts.max() if match_rts.size > 0 else -1
        rt_array = np.full(max_rt_idx + 1, -1, dtype=int)
        for i, rt in enumerate(match_rts):
            rt_array[rt] = i
        FitnessEvaluator.match_roundtypes = match_rts
        FitnessEvaluator.rt_array = rt_array

    def evaluate_population(self, pop_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an entire population of schedules.

        Args:
            pop_array (np.ndarray): Shape (pop_size, num_events). The core data.

        Returns:
            np.ndarray: Final fitness scores for the population. Shape (pop_size, num_objectives).
            np.ndarray: All team scores for the population. Shape (pop_size, num_teams, num_objectives).

        """
        # Dims for reference
        n_pop, _ = pop_array.shape

        # Preallocate arrays
        team_fitnesses = np.zeros((n_pop, self.n_teams, self.n_objs), dtype=float)

        # Get team-events mapping for the entire population
        valid_events_mask, team_events_pop = self.get_team_events(pop_array)

        # Slice event properties
        starts = self.event_properties.start[team_events_pop]
        stops = self.event_properties.stop[team_events_pop]
        loc_ids = self.event_properties.loc_idx[team_events_pop]
        paired_evt_ids = self.event_properties.paired_idx[team_events_pop]
        roundtype_ids = self.event_properties.roundtype_idx[team_events_pop]

        # Invalidate data for invalid events
        starts[~valid_events_mask] = self.max_int
        stops[~valid_events_mask] = self.max_int
        loc_ids[~valid_events_mask] = self.min_int
        paired_evt_ids[~valid_events_mask] = self.min_int
        roundtype_ids[~valid_events_mask] = self.min_int

        # Calculate scores for each objective
        team_fitnesses[:, :, 0] = self.score_break_time(starts, stops)
        team_fitnesses[:, :, 1] = self.score_loc_consistency(loc_ids, roundtype_ids)
        team_fitnesses[:, :, 2] = self.score_opp_variety(paired_evt_ids, pop_array)

        # Aggregate team scores into schedule scores
        min_s = team_fitnesses.min(axis=1)
        mean_s = team_fitnesses.mean(axis=1)
        mean_s = (mean_s * 0.5) + (min_s * 0.5)
        mean_s[mean_s == 0] = self._epsilon

        stddev_s = team_fitnesses.std(axis=1)
        coeff_s = stddev_s / mean_s
        vari_s = 1.0 / (1.0 + coeff_s)

        ptp = np.ptp(team_fitnesses, axis=1)
        range_s = 1.0 / (1.0 + ptp)

        mw, vw, rw = self.config.weights
        schedule_fitnesses = (mean_s * mw) + (vari_s * vw) + (range_s * rw)

        return schedule_fitnesses, team_fitnesses

    @classmethod
    def get_team_events(cls, pop_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Invert the (event -> team) mapping to a (team -> events) mapping for the entire population."""
        n_pop, _ = pop_array.shape

        # Preallocate the team-events array
        team_events_pop = np.full((n_pop, cls.n_teams, cls.max_events_per_team), -1, dtype=int)

        # Get indices of scheduled events
        sched_indices, event_indices = np.nonzero(pop_array >= 0)
        team_indices = pop_array[sched_indices, event_indices]

        # Handle the case with no scheduled events
        if sched_indices.size == 0:
            valid_events_mask = np.zeros_like(team_events_pop, dtype=bool)
            team_events_pop.fill(0)
            return valid_events_mask, team_events_pop

        # Create unique keys for (pop, team) pairs
        keys = (sched_indices * cls.n_teams) + team_indices

        # Count occurrences of each (pop, team) pair to determine group sizes
        counts = np.bincount(keys, minlength=n_pop * cls.n_teams)

        # Sort event indices by (pop, team)
        order = np.argsort(keys)
        sorted_event_indices = event_indices[order]

        # Compute group starts for each (pop, team)
        group_starts = np.zeros_like(counts, dtype=int)
        group_starts[1:] = np.cumsum(counts[:-1])

        # Compute within-group indices
        repeated_starts = np.repeat(group_starts, counts)
        within_group_indices = np.arange(sorted_event_indices.size, dtype=int) - repeated_starts

        # Map back to original indices
        sorted_keys = keys[order]
        pop_indices_sorted = sorted_keys // cls.n_teams
        team_indices_sorted = sorted_keys % cls.n_teams

        # Filter to only valid slots within max_events_per_team
        valid_mask = within_group_indices < cls.max_events_per_team
        pop_idx_final = pop_indices_sorted[valid_mask]
        team_idx_final = team_indices_sorted[valid_mask]
        slot_idx_final = within_group_indices[valid_mask]
        event_idx_final = sorted_event_indices[valid_mask]
        team_events_pop[pop_idx_final, team_idx_final, slot_idx_final] = event_idx_final

        # Invalidate data for invalid events
        valid_events_mask = team_events_pop >= 0
        team_events_pop[~valid_events_mask] = 0

        return valid_events_mask, team_events_pop

    def score_break_time(self, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
        """Vectorized break time scoring."""
        # Sort events by start time
        order = np.argsort(starts, axis=2)
        starts_sorted = np.take_along_axis(starts, order, axis=2)
        stops_sorted = np.take_along_axis(stops, order, axis=2)

        # Calculate breaks between consecutive events
        start_next = starts_sorted[..., 1:]
        stop_curr = stops_sorted[..., :-1]

        # Valid consecutive events must have valid start and stop times
        valid_mask = (start_next < self.max_int) & (stop_curr < self.max_int)

        # Calculate break durations in minutes
        breaks_seconds = start_next - stop_curr
        breaks_minutes = breaks_seconds / 60

        # Identify overlaps
        overlap_mask = (breaks_minutes < 0).any(axis=2, where=valid_mask)

        # Calculate mean
        count = valid_mask.sum(axis=2, dtype=float)
        count[count == 0] = self._epsilon

        mean_break = breaks_minutes.sum(axis=2, where=valid_mask) / count
        mean_break[mean_break <= 0] = self._epsilon

        # Calculate standard deviation
        diff_sq: np.ndarray = np.square(breaks_minutes - mean_break[..., np.newaxis])
        variance = diff_sq.sum(axis=2, where=valid_mask) / count
        std_dev: np.ndarray = np.sqrt(variance)

        # Calculate coefficient of variation
        coeff = std_dev / mean_break
        ratio = 1.0 / (1.0 + coeff)

        # Apply minimum break penalty
        threshold = 12  # minutes
        minbreak_count = ((breaks_minutes < threshold) & valid_mask).sum(axis=2)
        minbreak_penalty = (self._penalty * 100) ** minbreak_count

        # Apply penalties for zero breaks
        zeros_count = (breaks_minutes == 0).sum(axis=2, where=valid_mask)
        zeros_penalty = 0**zeros_count

        # Apply penalties
        final_scores = ratio * zeros_penalty * minbreak_penalty
        final_scores[mean_break == 0] = 0.0
        final_scores[overlap_mask] = 0.0

        return final_scores / (self.benchmark.best_timeslot_score or 1.0)

    @classmethod
    def score_loc_consistency(cls, loc_ids: np.ndarray, roundtype_ids: np.ndarray) -> np.ndarray:
        """Calculate location consistency score, prioritizing inter-round over intra-round consistency."""
        n_pop, n_teams, _ = loc_ids.shape
        match_roundtypes = cls.match_roundtypes
        n_match_rt = len(match_roundtypes)

        # Consistency score is only meaningful with 2+ match round types
        if n_match_rt < 2:
            return np.ones((n_pop, n_teams), dtype=float)

        # Create a (pop, team, rt, loc) boolean mask
        max_loc_idx = loc_ids.max()
        # No locations scheduled
        if max_loc_idx < 0:
            return np.ones((n_pop, n_teams), dtype=float)

        # match_rt_mask = np.isin(roundtype_ids, match_roundtypes) & (loc_ids >= 0)
        max_rt_id = max(roundtype_ids.max(), match_roundtypes.max())
        is_match_rt_lookup = np.zeros(max_rt_id + 1, dtype=bool)
        is_match_rt_lookup[match_roundtypes] = True
        match_rt_mask = is_match_rt_lookup[roundtype_ids] & (loc_ids >= 0)

        pop_indices, team_indices, _ = match_rt_mask.nonzero()
        loc_vals = loc_ids[match_rt_mask]
        rt_values = roundtype_ids[match_rt_mask]
        mapped_rt_indices = cls.rt_array[rt_values]

        # Inter-Round Consistency
        rt_loc_counts = np.zeros((n_pop, n_teams, n_match_rt, max_loc_idx + 1), dtype=int)
        # np.add.at(rt_loc_counts, (pop_indices, team_indices, mapped_rt_indices, loc_vals), 1)
        rt_loc_counts[pop_indices, team_indices, mapped_rt_indices, loc_vals] = 1

        # A team participated in a round type if its location counts for that RT are > 0.
        participated_in_rt_counts = rt_loc_counts.sum(axis=3, dtype=int)
        participated_in_rt = participated_in_rt_counts > 0

        # A location is in the intersection if its count across RTs equals the number of participated RTs.
        num_participated_rts = participated_in_rt.sum(axis=2, dtype=float)

        # Create a boolean mask of used locations (count > 0)
        loc_used_in_rt_mask = rt_loc_counts > 0

        # The result is the number of different round types a location was used in.
        loc_usage_across_rts = loc_used_in_rt_mask.sum(axis=2, dtype=float)

        # A location is in intersection if used in number of RTs equal to total number of RTs team participated in.
        intersection_mask = loc_usage_across_rts == num_participated_rts[..., np.newaxis]
        intersection_size = intersection_mask.sum(axis=2)

        # The union is the count of locations used in at least one round type.
        union_mask = loc_usage_across_rts > 0
        union_size = union_mask.sum(axis=2)

        # Handle the zero-division case explicitly.
        inter_round_scores = np.ones((n_pop, n_teams), dtype=float)
        valid_union = union_size > 0
        inter_round_scores[valid_union] = intersection_size[valid_union] / union_size[valid_union]

        # Intra-Round Consistency
        unique_locs_per_rt: np.ndarray = (rt_loc_counts > 0).sum(axis=3, dtype=float)
        unique_locs_per_rt[unique_locs_per_rt == 0] = cls._epsilon

        scores_per_rt = 1.0 / unique_locs_per_rt
        scores_per_rt[~participated_in_rt] = 1.0

        # Handle the zero-division case explicitly.
        intra_round_scores = np.ones((n_pop, n_teams), dtype=float)
        valid_num_rts = num_participated_rts > 0
        intra_round_scores[valid_num_rts] = (
            scores_per_rt.sum(axis=2)[valid_num_rts] / num_participated_rts[valid_num_rts]
        )

        # Final Combination
        total_matches_per_team = match_rt_mask.sum(axis=2)
        final_scores = (inter_round_scores * cls.loc_weight_rounds_inter) + (
            intra_round_scores * cls.loc_weight_rounds_intra
        )
        final_scores[total_matches_per_team <= 1] = 1.0

        return final_scores

    def score_opp_variety(self, paired_evt_ids: np.ndarray, pop_array: np.ndarray) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        n_pop, _ = pop_array.shape

        # Create a mask for valid opponent IDs
        valid_opp = paired_evt_ids >= 0

        # Invalidate opponent IDs for invalid events
        paired_evt_ids[~valid_opp] = 0
        schedule_indices = np.arange(n_pop, dtype=int)[:, None, None]

        # Get opponents for each schedule
        opponents = pop_array[schedule_indices, paired_evt_ids]
        opponents[~valid_opp] = self.max_int
        opponents.sort(axis=2)

        # Changes between consecutive opponents
        valid_mask = opponents[:, :, :-1] >= 0
        changes = np.diff(opponents, axis=2) != 0
        unique_counts = (changes & valid_mask).sum(axis=2)

        return self.benchmark.opponents[unique_counts]
