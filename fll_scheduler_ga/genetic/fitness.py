"""Fitness evaluator for the FLL Scheduler GA."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import EPSILON, FitnessObjective

if TYPE_CHECKING:
    from ..config.benchmark import FitnessBenchmark
    from ..config.ga_context import GaContext
    from ..data_model.config import TournamentConfig
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

        return not any(schedule.team_rounds_needed(team) for team in schedule.teams)


@dataclass(slots=True)
class FitnessEvaluator:
    """Calculates the fitness of a schedule."""

    config: TournamentConfig
    benchmark: FitnessBenchmark
    objectives: list[FitnessObjective] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        self.objectives.extend(list(FitnessObjective))

    def evaluate_population(self, population_array: np.ndarray, context: GaContext) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an entire population of schedules.

        Args:
            population_array (np.ndarray): Shape (pop_size, num_events). The core data.
            context (GaContext): Holds static lookup arrays like event_properties.

        Returns:
            np.ndarray: Final fitness scores for the population. Shape (pop_size, num_objectives).
            np.ndarray: All team scores for the population. Shape (pop_size, num_teams, num_objectives).

        """
        # Dims for reference
        n_pop = population_array.shape[0]
        n_teams = self.config.num_teams
        n_objs = len(self.objectives)
        # Preallocate arrays
        team_fitnesses = np.zeros((n_pop, n_teams, n_objs), dtype=float)
        # Get team-events mapping for the entire population
        # Shape: (n_pop, n_teams, max_events_per_team)
        team_events_pop = self.get_team_events_from_population(population_array, context)
        # Prepare lookup arrays
        valid_events_mask, lookup_events = self.prepare_lookup(team_events_pop)
        # Calculate scores for each objective
        event_props = context.event_properties
        team_fitnesses[:, :, 0] = self.calc_break_time_scores(
            event_props,
            valid_events_mask,
            lookup_events,
        )
        team_fitnesses[:, :, 1] = self.calc_location_consistency_scores(
            event_props,
            valid_events_mask,
            lookup_events,
        )
        team_fitnesses[:, :, 2] = self.calc_opponent_variety_scores(
            event_props,
            valid_events_mask,
            lookup_events,
            team_events_pop,
            population_array,
        )
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

    def get_team_events_from_population(self, population_array: np.ndarray, context: GaContext) -> np.ndarray:
        """Invert the (event -> team) mapping to a (team -> events) mapping for the entire population."""
        n_pop = population_array.shape[0]
        n_teams = self.config.num_teams
        max_events_per_team = context.max_events_per_team
        team_events_pop = np.full((n_pop, n_teams, max_events_per_team), -1, dtype=int)
        sched_indices, event_indices = np.where(population_array >= 0)
        team_indices = population_array[sched_indices, event_indices]
        if sched_indices.size > 0:
            keys = (sched_indices * n_teams) + team_indices
            order = np.lexsort((event_indices, keys))
            keys_sorted = keys[order]
            group_starts = np.r_[0, np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1]
            group_lengths = np.diff(np.r_[group_starts, keys_sorted.size])
            within_sorted = np.arange(keys_sorted.size, dtype=int) - np.repeat(group_starts, group_lengths)
            within = np.empty_like(within_sorted)
            within[order] = within_sorted
            valid_mask = within < max_events_per_team
            if np.any(valid_mask):
                team_events_pop[
                    sched_indices[valid_mask],
                    team_indices[valid_mask],
                    within[valid_mask],
                ] = event_indices[valid_mask]
        return team_events_pop

    def prepare_lookup(self, team_events_pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prepare lookup arrays for fitness calculations."""
        valid_events_mask = team_events_pop >= 0
        lookup_events = np.where(valid_events_mask, team_events_pop, 0)
        return valid_events_mask, lookup_events

    def calc_break_time_scores(
        self,
        event_properties: np.ndarray,
        valid_events_mask: np.ndarray,
        lookup_events: np.ndarray,
    ) -> np.ndarray:
        """Vectorized break time scoring."""
        max_int = np.iinfo(np.int64).max
        # Get start and stop times for each event
        starts = event_properties[lookup_events]["start"]
        stops = event_properties[lookup_events]["stop"]
        # Mask invalids with max int
        starts[~valid_events_mask] = max_int
        stops[~valid_events_mask] = max_int
        # Sort events by start time
        order = np.argsort(starts, axis=2)
        starts_sorted = np.take_along_axis(starts, order, axis=2)
        stops_sorted = np.take_along_axis(stops, order, axis=2)
        # Calculate breaks between consecutive events
        start_next = starts_sorted[:, :, 1:]
        stop_curr = stops_sorted[:, :, :-1]
        # Valid consecutive events must have valid start and stop times
        valid_consecutive = (start_next < max_int) & (stop_curr < max_int)

        breaks_seconds = start_next - stop_curr
        breaks_minutes = breaks_seconds / 60.0
        breaks_minutes[~valid_consecutive] = np.nan

        # Identify overlaps
        overlap_mask = np.any(breaks_minutes < 0, axis=2)

        mean_break = np.nan_to_num(
            np.nanmean(breaks_minutes, axis=2),
            nan=0.0,
        )
        std_dev = np.nan_to_num(
            np.nanstd(breaks_minutes, axis=2),
            nan=0.0,
        )

        coefficient = std_dev / (mean_break + EPSILON)
        ratio = 1.0 / (1.0 + coefficient)

        zeros = np.sum(breaks_minutes == 0, axis=2)
        b_penalty = self.benchmark.penalty**zeros

        final_scores = ratio * b_penalty
        final_scores[overlap_mask] = -1.0
        final_scores /= self.benchmark.best_timeslot_score
        return final_scores

    def calc_location_consistency_scores(
        self,
        event_properties: np.ndarray,
        valid_events_mask: np.ndarray,
        lookup_events: np.ndarray,
    ) -> np.ndarray:
        """Vectorized location consistency scoring."""
        locations = event_properties[lookup_events]["loc_idx"]
        locations[~valid_events_mask] = -1

        loc_sorted = np.sort(locations, axis=2)

        valid_mask = loc_sorted[:, :, :-1] >= 0
        changes = np.diff(loc_sorted, axis=2) != 0
        meaningful_changes = np.sum(changes & valid_mask, axis=2)

        has_at_least_one = loc_sorted[:, :, 0] >= 0
        unique_counts = np.where(has_at_least_one, meaningful_changes, 0)

        return self.benchmark.locations[unique_counts]

    def calc_opponent_variety_scores(
        self,
        event_properties: np.ndarray,
        valid_events_mask: np.ndarray,
        lookup_events: np.ndarray,
        team_events_pop: np.ndarray,
        population_array: np.ndarray,
    ) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        pop_size = team_events_pop.shape[0]

        paired_event_ids = event_properties[lookup_events]["paired_idx"]
        paired_event_ids[~valid_events_mask] = -1

        valid_opp = paired_event_ids >= 0
        lookup_opp_events = np.where(valid_opp, paired_event_ids, 0)

        schedule_indices = np.arange(pop_size)[:, None, None]
        opponents = population_array[schedule_indices, lookup_opp_events]
        opponents[~valid_opp] = -1

        opponents_sorted = np.sort(opponents, axis=2)

        valid_mask = opponents_sorted[:, :, :-1] >= 0
        changes = np.diff(opponents_sorted, axis=2) != 0
        meaningful_changes = np.sum(changes & valid_mask, axis=2)

        has_at_least_one = opponents_sorted[:, :, 1] >= 0
        unique_counts = np.where(has_at_least_one, meaningful_changes + 1, 0)

        return self.benchmark.opponents[unique_counts]
