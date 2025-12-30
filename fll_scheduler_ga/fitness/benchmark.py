"""Benchmarking module for fitness evaluation in tournament scheduling."""

from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.constants import EPSILON
from .benchmark_repository import BenchmarkSeedData

if TYPE_CHECKING:
    from ..config.app_schemas import TournamentConfig
    from ..config.pydantic_schemas import FitnessModel
    from ..data_model.event import EventFactory
    from .benchmark_repository import BenchmarkRepository

logger = getLogger(__name__)
BENCHMARKS_CACHE = Path(".benchmarks_cache/").resolve()
FITNESS_MODEL_VERSION = 3


@dataclass(slots=True)
class FitnessBenchmark:
    """Benchmark for evaluating fitness scores."""

    config: TournamentConfig
    model: FitnessModel
    repository: BenchmarkRepository
    opponent_benchmarker: FitnessBenchmarkOpponent
    breaktime_benchmarker: FitnessBenchmarkBreaktime
    flush_benchmarks: bool

    opponents: np.ndarray = field(init=False)
    best_timeslot_score: float = field(init=False)

    def run(self) -> None:
        """Run the fitness benchmarking process."""
        loaded_data = None
        if not self.flush_benchmarks:
            loaded_data = self.repository.load()

        if loaded_data and self._validate_data(loaded_data):
            self.opponents = loaded_data.opponents
            self.best_timeslot_score = loaded_data.best_timeslot_score
        else:
            logger.info("Calculating new benchmarks...")
            self.opponents = self.opponent_benchmarker.benchmark()
            self.best_timeslot_score = self.breaktime_benchmarker.benchmark()
            self.save_benchmarks()

    def _validate_data(self, data: BenchmarkSeedData) -> bool:
        """Validate loaded benchmark data."""
        if data.version != FITNESS_MODEL_VERSION:
            logger.warning(
                "Benchmark version mismatch: Expected %d, found %d. Recalculating benchmarks.",
                FITNESS_MODEL_VERSION,
                data.version,
            )
            return False
        return True

    def save_benchmarks(self) -> None:
        """Save the current benchmarks via repository."""
        data = BenchmarkSeedData(
            version=FITNESS_MODEL_VERSION,
            opponents=self.opponents,
            best_timeslot_score=self.best_timeslot_score,
        )
        self.repository.save(data)


@dataclass(slots=True)
class StableConfigHash:
    """Generate a stable hash for a given tournament configuration."""

    config: TournamentConfig
    model: FitnessModel

    def generate_hash(self) -> int:
        """Generate a stable hash for the parts of the config that define the benchmark."""
        # Create a tuple representation of relevant config parts
        config_representation = (
            self.config.get_canonical_round_tuples(),
            self.config.get_canonical_roundreqs_tuple(),
            self.model.minbreak_target,
            self.model.minbreak_penalty,
            self.model.zeros_penalty,
            self.config.num_teams,
        )

        # Using hashlib over built-in hash for stability
        return int(hashlib.sha256(str(config_representation).encode()).hexdigest(), 16)


@dataclass(slots=True)
class FitnessBenchmarkObjective(ABC):
    """Abstract base class for fitness benchmark objective."""

    config: TournamentConfig
    event_factory: EventFactory

    @abstractmethod
    def benchmark(self) -> Any:
        """Run the specific benchmark. To be implemented by subclasses."""


@dataclass(slots=True)
class FitnessBenchmarkOpponent(FitnessBenchmarkObjective):
    """Benchmark for opponent variety fitness."""

    def benchmark(self) -> np.ndarray:
        """Run the opponent variety fitness benchmarking."""
        logger.info("Running opponent variety benchmarks...")
        logger.debug("Finding events per round type:")

        max_matches_possible = 0
        max_matches_required = 0
        non_matches_required = 0
        round_str_to_idx = {r.roundtype: r.roundtype_idx for r in self.config.rounds}
        round_idx_to_rt = {v: k for k, v in round_str_to_idx.items()}
        for rt, events in self.event_factory.as_roundtypes().items():
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
class FitnessBenchmarkBreaktime(FitnessBenchmarkObjective):
    """Benchmark for break time consistency fitness."""

    model: FitnessModel

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

        minbreak_count = (breaks_active_minutes < self.model.minbreak_target).sum(axis=1)
        where_breaks_lt_target = (breaks_active_minutes < self.model.minbreak_target) & (breaks_active_minutes > 0)
        max_diff_breaktimes = np.zeros_like(minbreak_count)
        if where_breaks_lt_target.any():
            diffs = self.model.minbreak_target - breaks_active_minutes
            diffs[~where_breaks_lt_target] = 0.0
            max_diff_breaktimes = diffs.max(axis=1) / self.model.minbreak_target
        minbreak_exp = minbreak_count + max_diff_breaktimes
        minbreak_penalty = self.model.minbreak_penalty**minbreak_exp

        zeros_count = (breaks_cycle_minutes == 0).sum(axis=1)
        zeros_penalty = self.model.zeros_penalty**zeros_count

        final_scores = ratio * zeros_penalty * minbreak_penalty
        final_scores[mean_break_zero_mask] = 0.0
        final_scores[overlap_mask] = 0.0

        final_scores = final_scores[non_overlap_mask]
        indices = indices[non_overlap_mask]

        return final_scores, indices
