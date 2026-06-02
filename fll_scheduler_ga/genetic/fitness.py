"""Base class for fitness evaluators."""

import itertools
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import EPSILON, FitnessObjective

if TYPE_CHECKING:
    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TournamentConfig


logger = getLogger(__name__)


@dataclass(slots=True)
class FitnessEvaluator:
    """Base class for fitness evaluators."""

    # Configurations
    config: TournamentConfig
    event_properties: EventProperties
    benchmark_opponents: np.ndarray
    benchmark_best_timeslot_score: float
    loc_weight_rounds_inter: float
    loc_weight_rounds_intra: float
    agg_weights: tuple[float, ...]
    min_fitness_weight: float
    obj_weights: np.ndarray
    minbreak_target: int
    minbreak_penalty: float
    zeros_penalty: float
    # Globals
    max_int: int = np.iinfo(np.int64).max
    n_objectives: int = len(tuple(FitnessObjective))
    epsilon: float = EPSILON
    # TournamentConfig
    n_teams: int = field(init=False)
    n_max_events: int = field(init=False)
    n_match_rt: int = field(init=False)
    n_single_rt: int = field(init=False)
    single_roundtypes: np.ndarray = field(init=False)
    match_roundtypes: np.ndarray = field(init=False)
    rt_array: np.ndarray = field(init=False)

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

    def evaluate(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an entire population of schedules.

        Args:
            arr (np.ndarray): Shape (pop_size, num_events). The core data.

        Returns:
            np.ndarray: Final fitness scores for the population. Shape (pop_size, num_objectives).
            np.ndarray: All team scores for the population. Shape (pop_size, num_teams, num_objectives).

        """
        # Get team-events mapping for the entire population
        team_events = self.get_team_events(arr)
        # Slice event properties
        starts, stops_active, stops_cycle, loc_ids, paired_evt_ids, roundtype_ids = self._slice_event_properties(
            team_events=team_events
        )
        # Preallocate arrays
        team_fits_shape = (arr.shape[0], self.n_teams, self.n_objectives)
        team_fitnesses = np.zeros(team_fits_shape, dtype=float)
        # Calculate scores for each objective
        team_fitnesses[:, :, 0] = self.score_break_time(starts, stops_active, stops_cycle)
        team_fitnesses[:, :, 1] = self.score_loc_consistency(loc_ids, roundtype_ids)
        team_fitnesses[:, :, 2] = self.score_opp_variety(paired_evt_ids, arr)
        # Aggregate team scores into schedule scores
        schedule_fitnesses = self._aggregate_team_scores(team_fitnesses, team_axis=1)
        return schedule_fitnesses, team_fitnesses

    def get_team_events(self, arr: np.ndarray) -> np.ndarray:
        """Invert the (event -> team) mapping to a (team -> events) mapping for the entire population."""
        # Preallocate the team-events array
        n_pop = arr.shape[0]
        team_events = np.full((n_pop, self.n_teams, self.n_max_events), -1, dtype=int)
        # Get indices of scheduled events
        sched_indices, event_indices = (arr >= 0).nonzero()
        # Handle the case with no scheduled events
        if sched_indices.size == 0:
            team_events.fill(0)
            return team_events
        # Create unique keys for (pop, team) pairs
        team_indices = arr[sched_indices, event_indices]
        keys = (sched_indices * self.n_teams) + team_indices
        # Count occurrences of each (pop, team) pair to determine group sizes
        counts = np.bincount(keys, minlength=n_pop * self.n_teams)
        # Sort event indices by (pop, team)
        order = keys.argsort()
        sorted_event_indices = event_indices[order]
        # Compute group starts for each (pop, team)
        group_starts = np.zeros_like(counts, dtype=int)
        group_starts[1:] = counts[:-1].cumsum()
        # Compute within-group indices
        repeated_starts = group_starts.repeat(counts)
        within_group_indices = np.arange(sorted_event_indices.size, dtype=int) - repeated_starts
        # Map back to original indices
        sorted_keys = keys[order]
        pop_indices_sorted = sorted_keys // self.n_teams
        team_indices_sorted = sorted_keys % self.n_teams
        # Filter to only valid slots within max_events_per_team
        valid_mask = within_group_indices < self.n_max_events
        pop_idx_final = pop_indices_sorted[valid_mask]
        team_idx_final = team_indices_sorted[valid_mask]
        slot_idx_final = within_group_indices[valid_mask]
        event_idx_final = sorted_event_indices[valid_mask]
        team_events[pop_idx_final, team_idx_final, slot_idx_final] = event_idx_final
        return team_events

    def score_break_time(self, starts: np.ndarray, stops_active: np.ndarray, stops_cycle: np.ndarray) -> np.ndarray:
        """Vectorized break time scoring."""
        # Sort events by start time
        order = starts.argsort(axis=2)
        starts_sorted = np.take_along_axis(starts, order, axis=2)
        stops_active_sorted = np.take_along_axis(stops_active, order, axis=2)
        stops_cycle_sorted = np.take_along_axis(stops_cycle, order, axis=2)
        # Calculate breaks between consecutive events
        start_next = starts_sorted[:, :, 1:]
        stop_active_curr = stops_active_sorted[:, :, :-1]
        stop_cycle_curr = stops_cycle_sorted[:, :, :-1]
        # Calculate break durations in minutes
        breaks_active_minutes = np.subtract(start_next, stop_active_curr) / 60
        breaks_cycle_minutes = np.subtract(start_next, stop_cycle_curr) / 60
        # Identify overlaps
        overlap_mask = (breaks_cycle_minutes < 0).any(axis=2)
        # Calculate mean
        valid_mask = breaks_cycle_minutes >= 0
        count = valid_mask.sum(axis=2, dtype=int)
        mean_break = breaks_cycle_minutes.sum(axis=2) / count
        mean_break_zero_mask = mean_break == 0
        mean_break[mean_break_zero_mask] = self.epsilon
        # Calculate standard deviation
        diff_sq: np.ndarray = np.square(breaks_cycle_minutes - mean_break[:, :, np.newaxis])
        variance = diff_sq.sum(axis=2) / count
        std_dev: np.ndarray = np.sqrt(variance)
        # Calculate coefficient of variation
        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)
        # Apply minimum break penalty
        minbreak_count = (breaks_active_minutes < self.minbreak_target).sum(axis=2)
        where_breaks_lt_target = (breaks_active_minutes < self.minbreak_target) & (breaks_active_minutes > 0)
        max_diff_breaktimes = np.zeros_like(minbreak_count)
        if where_breaks_lt_target.any():
            diffs = self.minbreak_target - breaks_active_minutes
            diffs[~where_breaks_lt_target] = 0.0
            max_diff_breaktimes = diffs.max(axis=2) / self.minbreak_target
        minbreak_exp = minbreak_count + max_diff_breaktimes
        minbreak_penalty = self.minbreak_penalty**minbreak_exp
        # Apply penalties for zero breaks
        zeros_count = (breaks_cycle_minutes == 0).sum(axis=2)
        zeros_penalty = self.zeros_penalty**zeros_count
        # Apply penalties
        final_scores = ratio * zeros_penalty * minbreak_penalty
        final_scores[mean_break_zero_mask | overlap_mask] = 0.0
        return final_scores / self.benchmark_best_timeslot_score

    def score_loc_consistency(self, loc_ids: np.ndarray, roundtype_ids: np.ndarray) -> np.ndarray:
        """Calculate location consistency score, prioritizing inter-round over intra-round consistency."""
        n_pop, n_teams, _ = loc_ids.shape
        shape = (n_pop, n_teams)
        # Consistency score is only meaningful with 1+ match round types
        if self.n_match_rt < 1:
            return np.ones(shape, dtype=float)
        # Create a (pop, team, rt, loc) boolean mask
        max_loc_idx = loc_ids.max()
        # No locations scheduled
        if max_loc_idx < 0:
            return np.ones(shape, dtype=float)
        max_rt_id = max(roundtype_ids.max(), self.match_roundtypes.max())
        is_match_rt_lookup = np.zeros(max_rt_id + 1, dtype=bool)
        is_match_rt_lookup[self.match_roundtypes] = True
        match_rt_mask = is_match_rt_lookup[roundtype_ids] & (loc_ids >= 0)
        pop_indices, team_indices, _ = match_rt_mask.nonzero()
        loc_vals = loc_ids[match_rt_mask]
        rt_values = roundtype_ids[match_rt_mask]
        mapped_rt_indices = self.rt_array[rt_values]
        # Inter-Round Consistency
        inter_round_scores = np.ones(shape, dtype=float)
        # Build count array: (pop, team, rt, loc)
        rt_loc_counts = np.zeros((n_pop, n_teams, self.n_match_rt, max_loc_idx + 1), dtype=int)
        rt_loc_counts[pop_indices, team_indices, mapped_rt_indices, loc_vals] = 1
        # A team participated in a round type if its location counts for that RT are > 0.
        participated_in_rt_counts = rt_loc_counts.sum(axis=3, dtype=int)
        participated_in_rt = participated_in_rt_counts > 0
        # A location is in the intersection if its count across RTs equals the number of participated RTs.
        num_participated_rts = participated_in_rt.sum(axis=2, dtype=float)
        if self.n_match_rt >= 2:
            # Create a boolean mask of used locations (count > 0)
            loc_used_in_rt_mask = rt_loc_counts > 0
            # The result is the number of different round types a location was used in.
            loc_usage_across_rts = loc_used_in_rt_mask.sum(axis=2, dtype=float)
            # A location is in intersection if used in number of RTs equal to total number of RTs team participated in.
            intersection_mask = loc_usage_across_rts == num_participated_rts[:, :, np.newaxis]
            intersection_size = intersection_mask.sum(axis=2)
            # The union is the count of locations used in at least one round type.
            union_mask = loc_usage_across_rts > 0
            union_size = union_mask.sum(axis=2)
            # Handle the zero-division case explicitly.
            valid_union = union_size > 0
            inter_round_scores[valid_union] = intersection_size[valid_union] / union_size[valid_union]
        # Intra-Round Consistency
        unique_locs_per_rt: np.ndarray = (rt_loc_counts > 0).sum(axis=3, dtype=float)
        unique_locs_per_rt[unique_locs_per_rt == 0] = self.epsilon
        scores_per_rt = 1.0 / unique_locs_per_rt
        scores_per_rt[~participated_in_rt] = 1.0
        # Handle the zero-division case explicitly.
        valid_num_rts = num_participated_rts > 0
        intra_round_scores = np.ones(shape, dtype=float)
        intra_round_scores[valid_num_rts] = (
            scores_per_rt.sum(axis=2)[valid_num_rts] / num_participated_rts[valid_num_rts]
        )
        # Final Combination
        total_matches_per_team = match_rt_mask.sum(axis=2)
        final_scores = (inter_round_scores * self.loc_weight_rounds_inter) + (
            intra_round_scores * self.loc_weight_rounds_intra
        )
        final_scores[total_matches_per_team <= 1] = 1.0
        return final_scores

    def score_opp_variety(self, paired_evt_ids: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        n_pop, _ = arr.shape
        # Create a mask for valid opponent IDs
        valid_opp = paired_evt_ids >= 0
        invalid_opp = ~valid_opp
        # Invalidate opponent IDs for invalid events
        paired_evt_ids[invalid_opp] = 0
        schedule_indices = np.arange(n_pop, dtype=int)[:, None, None]
        # Get opponents for each schedule
        opponents = arr[schedule_indices, paired_evt_ids]
        opponents[invalid_opp] = self.max_int
        opponents.sort(axis=2)
        # Changes between consecutive opponents
        valid_mask = opponents[:, :, :-1] >= 0
        diffs = np.diff(opponents, axis=2)
        changes = diffs != 0
        # Check if there are single round types
        unique_counts = (changes & valid_mask).sum(axis=2)
        unique_counts = unique_counts + 1 if self.n_single_rt == 0 else unique_counts
        return self.benchmark_opponents[unique_counts]

    def _slice_event_properties(self, team_events: np.ndarray) -> tuple[np.ndarray, ...]:
        """Slice event properties arrays based on team events.

        Args:
            team_events: Array of event IDs for each team
        Returns:
            tuple[np.ndarray, ...]: Tuple of sliced event properties arrays.

        """
        return (
            self.event_properties.start[team_events],
            self.event_properties.stop_active[team_events],
            self.event_properties.stop_cycle[team_events],
            self.event_properties.loc_idx[team_events],
            self.event_properties.paired_idx[team_events],
            self.event_properties.roundtype_idx[team_events],
        )

    def _aggregate_team_scores(self, team_fitnesses: np.ndarray, team_axis: int) -> np.ndarray:
        """Aggregate team fitness scores into schedule fitness scores.

        Args:
            team_fitnesses: Array of team fitness scores
            team_axis: Axis along which teams are indexed
        Returns:
            np.ndarray: Aggregated schedule fitness scores.

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


@dataclass(slots=True)
class FitnessBenchmarkOpponent:
    """Benchmark for opponent variety fitness."""

    config: TournamentConfig
    event_repo: EventRepository

    def benchmark(self) -> np.ndarray:
        """Run the opponent variety fitness benchmarking."""
        logger.info("Running opponent variety benchmarks...")
        logger.debug("Finding events per round type:")
        max_matches_possible = 0
        max_matches_required = 0
        non_matches_required = 0
        round_str_to_idx = {r.roundtype: r.roundtype_idx for r in self.config.rounds}
        round_idx_to_rt = {v: k for k, v in round_str_to_idx.items()}
        for rt, events in self.event_repo.roundtypes.items():
            rti_to_rt = round_idx_to_rt[rt]
            roundreq = self.config.roundreqs[rti_to_rt]
            round_to_tpr = self.config.round_idx_to_tpr[rt]
            if round_to_tpr == 2:
                max_matches_possible += len(events)
                max_matches_required += roundreq
            elif round_to_tpr == 1:
                non_matches_required += roundreq
        num_matches_considered = max_matches_required + non_matches_required + 1
        cache_scorer = {}
        for k in range(num_matches_considered):
            cache_scorer[k] = 0.0
        for n_rounds in range(1, max_matches_required + 1):
            ratio = n_rounds / max_matches_possible
            cache_scorer[n_rounds] = 1 / (1 + ratio)
        if non_matches_required > 0:
            cache_scorer[max_matches_required + non_matches_required] = 0
        maximum_score = cache_scorer[1]
        minimum_score = cache_scorer[max_matches_required]
        diff = maximum_score - minimum_score
        if diff <= 0:
            diff = EPSILON
        raw_scores = tuple(cache_scorer.values())
        logger.debug("Raw location/opponent scores: %s", raw_scores)
        opponents = [abs((s - maximum_score) / diff) if s != 0 else 0 for s in raw_scores]
        opponents_arr = np.array(opponents, dtype=float)
        logger.debug("Opponent variety scores:")
        for k, v in enumerate(opponents_arr):
            logger.debug("  %d opponent(s): %.6f", k, v)
        if not opponents_arr.any():
            logger.warning("No valid schedules could be generated.")
            return np.array([])
        return opponents_arr


@dataclass(slots=True)
class FitnessBenchmarkBreaktime:
    """Benchmark for break time consistency fitness."""

    config: TournamentConfig
    minbreak_target: int
    minbreak: float
    zeros: float

    def benchmark(self) -> float:
        """Run the break time consistency fitness benchmarking."""
        logger.info("Running break time consistency benchmarks...")
        all_ts = self.config.all_timeslots
        all_starts = np.array([int(ts.start.timestamp()) for ts in all_ts], dtype=int)
        all_stops_active = np.array([int(ts.stop_active.timestamp()) for ts in all_ts], dtype=int)
        all_stops_cycle = np.array([int(ts.stop_cycle.timestamp()) for ts in all_ts], dtype=int)
        logger.debug("Finding timeslots per round type:")
        timeslots_by_round = {r.roundtype: [ts.idx for ts in r.timeslots] for r in self.config.rounds}
        # Generate intra-round combinations
        round_slot_combos = self.generate_intra_round_breaktime_combinations(timeslots_by_round)
        # Filter, score, and store valid schedules
        logger.debug("Generating and filtering all possible team schedules")
        raw_product = itertools.product(*round_slot_combos)  # Cartesian product of round combinations
        flattened_indices = [list(itertools.chain.from_iterable(p)) for p in raw_product]
        if not flattened_indices:
            logger.warning("No possible schedules could be generated.")
            return 0
        # Convert to 2D matrix (n_combinations, n_events)
        indices_matrix = np.array(flattened_indices, dtype=int)
        total_combinations = indices_matrix.shape[0]
        logger.debug("indices_matrix (Shape: %s):\n%s", indices_matrix.shape, indices_matrix)
        logger.debug("total_combinations: %d", total_combinations)
        logger.debug("calculating breaktime scores vectorized...")
        valid_scores, _ = self.score_breaktime(indices_matrix, all_starts, all_stops_active, all_stops_cycle)
        num_valid = valid_scores.shape[0]
        logger.debug("num_valid: %d", num_valid)
        if num_valid == 0:
            logger.warning("No valid schedules could be generated.")
            return 0
        best_timeslot_score = valid_scores.max()
        if best_timeslot_score == 0:
            best_timeslot_score = 1  # Avoid division by zero
        logger.debug("Best timeslot score: %f", best_timeslot_score)
        # Normalize
        normalized_scores = valid_scores / best_timeslot_score
        # Reporting
        unique_scores = Counter(normalized_scores)
        logger.debug("Unique scores found: %d", len(unique_scores))
        logger.debug("Number of best scores: %d", unique_scores.get(1.0, 0))
        most_common = unique_scores.most_common(50)
        for score, count in most_common:
            logger.debug("  Score %s: %d occurrences", f"{score:<.16f}", count)
        avg_score = sum(score for score, _ in most_common) / len(most_common)
        logger.debug("Average score of most common: %f", avg_score)
        return best_timeslot_score

    def generate_intra_round_breaktime_combinations(
        self, timeslots_by_round: dict[str, list[int]]
    ) -> list[tuple[tuple[int, ...], ...]]:
        """Generate all intra-round breaktime combinations."""
        logger.debug("Generating all possible schedules per round type:")
        round_slot_combos = []
        for rt, num_needed in self.config.roundreqs.items():
            timeslot_indices = timeslots_by_round.get(rt, [])
            combos = tuple(itertools.combinations(timeslot_indices, num_needed))
            round_slot_combos.append(combos)
            logger.debug("  roundtype: %s", rt)
            logger.debug("    %d timeslots", len(timeslot_indices))
            logger.debug("      timeslots: %s", timeslot_indices)
            logger.debug("    %d combinations", len(combos))
            logger.debug("      combinations: %s", combos)
        return round_slot_combos

    def score_breaktime(
        self, indices: np.ndarray, starts: np.ndarray, stops_active: np.ndarray, stops_cycle: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Calculate breaktime fitnesses vectorized."""
        starts = starts[indices]
        stops = stops_active[indices]
        stops_cycle = stops_cycle[indices]
        order = starts.argsort(axis=1)
        starts_sorted = np.take_along_axis(starts, order, axis=1)
        stops_active_sorted = np.take_along_axis(stops, order, axis=1)
        stops_cycle_sorted = np.take_along_axis(stops_cycle, order, axis=1)
        start_next = starts_sorted[:, 1:]
        stop_active_curr = stops_active_sorted[:, :-1]
        stop_cycle_curr = stops_cycle_sorted[:, :-1]
        breaks_active_seconds = start_next - stop_active_curr
        breaks_active_minutes = breaks_active_seconds / 60
        breaks_cycle_seconds = start_next - stop_cycle_curr
        breaks_cycle_minutes = breaks_cycle_seconds / 60
        overlap_mask = (breaks_cycle_minutes < 0).any(axis=1)
        non_overlap_mask = ~overlap_mask
        valid_mask = breaks_cycle_minutes >= 0
        count = valid_mask.sum(axis=1, dtype=int)
        mean_break = breaks_cycle_minutes.sum(axis=1) / count
        mean_break_zero_mask = mean_break == 0
        mean_break[mean_break_zero_mask] = EPSILON
        diff_sq: np.ndarray = np.square(breaks_cycle_minutes - mean_break[:, np.newaxis])
        variance = diff_sq.sum(axis=1) / count
        std_dev: np.ndarray = np.sqrt(variance)
        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)
        minbreak_count = (breaks_active_minutes < self.minbreak_target).sum(axis=1)
        where_breaks_lt_target = (breaks_active_minutes < self.minbreak_target) & (breaks_active_minutes > 0)
        max_diff_breaktimes = np.zeros_like(minbreak_count)
        if where_breaks_lt_target.any():
            diffs = self.minbreak_target - breaks_active_minutes
            diffs[~where_breaks_lt_target] = 0.0
            max_diff_breaktimes = diffs.max(axis=1) / self.minbreak_target
        minbreak_exp = minbreak_count + max_diff_breaktimes
        minbreak_penalty = self.minbreak**minbreak_exp
        zeros_count = (breaks_cycle_minutes == 0).sum(axis=1)
        zeros_penalty = self.zeros**zeros_count
        final_scores = ratio * zeros_penalty * minbreak_penalty
        final_scores[mean_break_zero_mask] = 0.0
        final_scores[overlap_mask] = 0.0
        final_scores = final_scores[non_overlap_mask]
        indices = indices[non_overlap_mask]
        return final_scores, indices
