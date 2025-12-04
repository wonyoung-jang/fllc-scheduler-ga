"""Benchmarking module for fitness evaluation in tournament scheduling."""

from __future__ import annotations

import hashlib
import itertools
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import BENCHMARKS_CACHE, EPSILON, FITNESS_MODEL_VERSION, FITNESS_PENALTY
from ..io.seed_fitness_benchmark import BenchmarkLoad, BenchmarkSave, BenchmarkSeedData

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.schemas import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties

logger = getLogger(__name__)


@dataclass(slots=True)
class FitnessBenchmark:
    """Benchmark for evaluating fitness scores."""

    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties

    penalty: float = FITNESS_PENALTY
    seed_file: Path = None
    opponents: np.ndarray = None
    flush_benchmarks: bool = False
    best_timeslot_score: float = None
    lookup_breaktime_keys: np.ndarray = None
    lookup_breaktime_values: np.ndarray = None

    def __post_init__(self) -> None:
        """Post-initialization to validate run benchmark."""
        self.run()

    def run(self) -> None:
        """Run the fitness benchmarking process."""
        self.init_cache_file()
        loaded = False
        if not self.flush_benchmarks and self.seed_file.exists():
            loaded = self.load_benchmarks()

        if not loaded:
            self.run_benchmarks()
            self.save_benchmarks()

    def init_cache_file(self) -> None:
        """Initialize the cache file path based on the configuration hash."""
        cache_dir = BENCHMARKS_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)
        config_hash = self._get_config_hash()
        self.seed_file = cache_dir / f"benchmark_cache_{config_hash}.pkl"

    def load_benchmarks(self) -> bool:
        """Load benchmarks from cache or run calculations if cache is invalid/missing."""
        try:
            logger.debug("Loading fitness benchmarks from cache: %s", self.seed_file)
            seed_benchmark_data = BenchmarkLoad(self.seed_file).load()
            if seed_benchmark_data.version != FITNESS_MODEL_VERSION:
                logger.warning(
                    "Benchmark data version mismatch: Expected (%d), found (%d). Recalculating benchmarks...",
                    FITNESS_MODEL_VERSION,
                    seed_benchmark_data.version,
                )
                return False
            self.opponents = seed_benchmark_data.opponents
            self.best_timeslot_score = seed_benchmark_data.best_timeslot_score
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.warning("Could not load cache file. Recalculating benchmarks.")
            self.seed_file.unlink(missing_ok=True)
            return False
        else:
            return True

    def run_benchmarks(self) -> None:
        """Run all fitness benchmark calculations."""
        self.run_benchmark_opponent()
        self.run_benchmark_break()

    def save_benchmarks(self) -> None:
        """Save the current benchmarks to cache."""
        seed_benchmark_data = BenchmarkSeedData(
            version=FITNESS_MODEL_VERSION,
            opponents=self.opponents,
            best_timeslot_score=self.best_timeslot_score,
        )
        saver = BenchmarkSave(
            path=self.seed_file,
            data=seed_benchmark_data,
        )
        saver.save()

    def _get_config_hash(self) -> int:
        """Generate a stable hash for the parts of the config that define the benchmark."""
        # Canonical representation of rounds
        round_tuples = tuple(
            (
                r.roundtype,
                r.roundtype_idx,
                r.rounds_per_team,
                r.teams_per_round,
                frozenset(r.times),
                r.start_time,
                r.stop_time,
                r.duration_minutes,
                r.location_type,
                frozenset(r.locations),
                r.num_timeslots,
                frozenset(r.timeslots),
                r.slots_total,
                r.slots_required,
                r.slots_empty,
                r.unfilled_allowed,
            )
            for r in self.config.rounds
        )

        # Canonical representation of requirements
        req_tuple = tuple(sorted(self.config.roundreqs.items()))

        # Include the penalty in the hash
        config_representation = (round_tuples, req_tuple, self.penalty, self.config.num_teams)

        # Using hashlib over built-in hash for stability
        return int(hashlib.sha256(str(config_representation).encode()).hexdigest(), 16)

    def run_benchmark_opponent(self) -> None:
        """Run the opponent variety fitness benchmarking."""
        logger.info("Running opponent variety benchmarks...")
        logger.debug("Finding events per round type:")

        max_matches_possible = 0
        max_matches_required = 0
        non_matches_required = 0
        round_idx_to_rt = {v: k for k, v in self.config.round_str_to_idx.items()}
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
        cache_scorer = dict.fromkeys(range(num_matches_considered), 0.0)
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
        self.opponents = np.array(opponents, dtype=float)

        logger.debug("Opponent variety scores:")
        for k, v in enumerate(self.opponents):
            logger.debug("  %d opponent(s): %.6f", k, v)

        if not self.opponents.any():
            logger.warning("No valid schedules could be generated.")
            return

    def run_benchmark_break(self) -> None:
        """Run the time slot fitness benchmarking."""
        logger.info("Running break time consistency benchmarks...")
        all_ts = self.config.all_timeslots
        all_starts = np.array([int(ts.start.timestamp()) for ts in all_ts], dtype=int)
        all_stops = np.array([int(ts.stop.timestamp()) for ts in all_ts], dtype=int)

        logger.debug("Finding timeslots per round type:")
        timeslots_by_round = {r.roundtype: [ts.idx for ts in r.timeslots] for r in self.config.rounds}
        for rt, timeslot_idx in timeslots_by_round.items():
            roundtype = f"{rt:<10}"
            n_timeslots = len(timeslot_idx)
            ts_idx_to_str = ", ".join(str(ts) for ts in timeslot_idx)
            logger.debug("  %s: %d unique timeslots\n    %s", roundtype, n_timeslots, ts_idx_to_str)

        # Generate intra-round combinations
        logger.debug("Generating all possible schedules per round type:")
        round_slot_combos = []
        for rt, num_needed in self.config.roundreqs.items():
            available_slots = timeslots_by_round.get(rt, [])
            combos = tuple(itertools.combinations(available_slots, num_needed))
            round_slot_combos.append(combos)
            logger.debug("  %s: %d round combinations", f"{rt:<10}", len(combos))
            logger.debug("    %s", ", ".join(str(combo) for combo in combos))

        # Filter, score, and store valid schedules
        logger.debug("Generating and filtering all possible team schedules")
        np.array(self.config.all_timeslots, dtype=object)
        total_combinations = 0

        raw_product = itertools.product(*round_slot_combos)  # Cartesian product of round combinations
        flattened_indices = [list(itertools.chain.from_iterable(p)) for p in raw_product]
        if not flattened_indices:
            logger.warning("No possible schedules could be generated.")
            return

        # Convert to 2D matrix (n_combinations, n_events)
        indices_matrix = np.array(flattened_indices, dtype=int)
        total_combinations = indices_matrix.shape[0]

        logger.debug("raw_product: %s", raw_product)
        logger.debug("flattened_indices: %s", flattened_indices)
        logger.debug("total_combinations: %d", total_combinations)
        logger.debug("calculating breaktime scores vectorized...")

        scores, valid_mask = self._score_breaktime_vectorized(indices_matrix, all_starts, all_stops)
        valid_indices = indices_matrix[valid_mask]
        valid_scores = scores[valid_mask]
        num_valid = valid_scores.shape[0]
        logger.debug("num_valid: %d", num_valid)
        if num_valid == 0:
            logger.warning("No valid schedules could be generated.")
            return

        self.best_timeslot_score = valid_scores.max()
        if self.best_timeslot_score == 0:
            self.best_timeslot_score = 1  # Avoid division by zero

        logger.debug("Best timeslot score: %f", self.best_timeslot_score)

        # Normalize
        normalized_scores = valid_scores / self.best_timeslot_score

        # Caching frozenset implementation
        cache_data = {frozenset(row): score for row, score in zip(valid_indices, normalized_scores, strict=True)}
        size_frozenset_cache = sys.getsizeof(cache_data)
        logger.debug("Size of frozenset cache: %d bytes", size_frozenset_cache)

        # Caching numpy array implementation
        # One should be able to index a set of indices and get a single score
        valid_indices.sort(axis=1)
        n_cols = valid_indices.shape[1]
        dtype_view = np.dtype((np.void, valid_indices.dtype.itemsize * n_cols))
        compact_keys = np.ascontiguousarray(valid_indices).view(dtype_view).ravel()

        sort_order = np.argsort(compact_keys)
        self.lookup_breaktime_keys = compact_keys[sort_order]
        self.lookup_breaktime_values = normalized_scores[sort_order]

        total_np_bytes = self.lookup_breaktime_keys.nbytes + self.lookup_breaktime_values.nbytes
        logger.debug("--- Memory Comparison ---")
        logger.debug("Python Dict (Struct only): %d bytes", size_frozenset_cache)
        logger.debug("NumPy Compact (Total):     %d bytes", total_np_bytes)

        # Reporting
        unique_scores = Counter(normalized_scores)
        logger.debug("Unique scores found: %d", len(unique_scores))

        most_common = unique_scores.most_common(50)
        for score, count in most_common:
            logger.debug("  Score %s: %d occurrences", f"{score:<.16f}", count)

        avg_score = sum(score for score, _ in most_common) / len(most_common)
        logger.debug("Average score of most common: %f", avg_score)

    def _score_breaktime_vectorized(
        self, indices: np.ndarray, all_starts: np.ndarray, all_stops: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Calculate breaktime fitnesses vectorized."""
        starts = all_starts[indices]
        stops = all_stops[indices]

        order = np.argsort(starts, axis=1)
        starts_sorted = np.take_along_axis(starts, order, axis=1)
        stops_sorted = np.take_along_axis(stops, order, axis=1)

        breaks_seconds = starts_sorted[:, 1:] - stops_sorted[:, :-1]
        breaks_minutes = breaks_seconds / 60

        has_overlap = np.any(breaks_minutes < 0, axis=1)
        valid_mask = ~has_overlap

        mean_break = breaks_minutes.mean(axis=1)
        mean_break[mean_break == 0] = EPSILON

        std_dev = breaks_minutes.std(axis=1)

        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)

        zeros_count = np.sum(breaks_minutes == 0, axis=1)
        zeros_penalty = self.penalty**zeros_count

        final_scores = ratio * zeros_penalty
        final_scores[has_overlap] = 0.0

        return final_scores, valid_mask
