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
    loc_weight_primary: ClassVar[float] = 0.9
    loc_weight_bonus: ClassVar[float] = 0.1
    loc_norm_max: ClassVar[float] = (1.0 * loc_weight_primary) + (0.5 * loc_weight_bonus)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives = list(FitnessObjective)
        self.max_events_per_team = self.config.max_events_per_team
        FitnessEvaluator.n_teams = self.config.num_teams
        FitnessEvaluator.n_objs = len(self.objectives)
        match_roundtypes = np.array([rt_idx for rt_idx, tpr in self.config.round_idx_to_tpr.items() if tpr == 2])
        FitnessEvaluator.match_roundtypes = match_roundtypes
        min_matches = (
            min(r.rounds_per_team for r in self.config.rounds if r.teams_per_round == 2)
            if len(match_roundtypes) > 0
            else 0
        )
        FitnessEvaluator.min_matches = min_matches

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
        n_pop, n_teams, _ = loc_ids.shape

        # Create mask for valid locations in match roundtypes
        match_roundtypes = FitnessEvaluator.match_roundtypes
        n_match_rt = len(match_roundtypes)

        if n_match_rt == 0:
            return np.ones((n_pop, n_teams), dtype=float)

        match_rt_mask = np.isin(roundtype_ids, match_roundtypes) & (loc_ids >= 0)

        # Total matches per team
        total_matches_per_team = match_rt_mask.sum(axis=2)

        # Get all unique locations per team
        max_loc_idx = loc_ids.max()

        # 3D mask: (n_pop, n_teams, location_idx)
        team_loc_mask = np.zeros((n_pop, n_teams, max_loc_idx + 1), dtype=bool)

        # Get indices of all valid match events for vectorized assignment
        pop_indices, team_indices, _ = np.where(match_rt_mask)
        loc_vals = loc_ids[match_rt_mask]

        # True where team has that location
        team_loc_mask[pop_indices, team_indices, loc_vals] = True

        # The number of unique locations is the sum along the location axis
        total_unique_locs = team_loc_mask.sum(axis=2)

        # Primary Score: fewer unique locations is better
        max_negative = total_matches_per_team - 1
        negative = total_unique_locs - 1
        primary_scores = 1.0 - (negative / np.maximum(max_negative, 1))
        primary_scores[max_negative <= 0] = 1.0  # If only one or zero matches, perfect score

        # Bonus Score: intersection across round types
        # Create a (pop, team, rt, loc) mask
        rt_map = {rt_val: i for i, rt_val in enumerate(match_roundtypes)}
        rt_values = roundtype_ids[match_rt_mask]
        mapped_rt_indices = np.array([rt_map[rt] for rt in rt_values])

        # A 4D mask for locations used per round type
        rt_loc_mask = np.zeros((n_pop, n_teams, n_match_rt, max_loc_idx + 1), dtype=bool)
        rt_loc_mask[pop_indices, team_indices, mapped_rt_indices, loc_vals] = True

        # For each team, determine which roundtypes they actually participate in.
        # Shape: (n_pop, n_teams, n_match_rts)
        participated_in_rt = np.any(rt_loc_mask, axis=3)
        condition = ~participated_in_rt[:, :, :, np.newaxis]
        x = True

        # Create eval mask where non-participated round types count as having all locations
        eval_mask = np.where(condition, x, rt_loc_mask)

        # np.all() along the round-type axis
        intersection_mask = np.all(eval_mask, axis=2)  # Shape: (n_pop, n_teams, max_loc)
        overlap_count = np.sum(intersection_mask, axis=2)

        # Normalize the bonus score
        min_matches = FitnessEvaluator.min_matches
        bonus_scores = overlap_count / min_matches if min_matches > 0 else np.ones((n_pop, n_teams), dtype=float)

        # Combine with weights
        wp = FitnessEvaluator.loc_weight_primary
        wb = FitnessEvaluator.loc_weight_bonus
        normmax = FitnessEvaluator.loc_norm_max
        return ((primary_scores * wp) + (bonus_scores * wb)) / normmax

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
