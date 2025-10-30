"""Time slot fitness benchmarking.

This module generates all possible valid time slot combinations for a single team
based on the tournament configuration and calculates a "break time fitness" score
for each combination. This helps identify the theoretically best and worst
schedules a team could receive, independent of other teams.
"""

from __future__ import annotations

import hashlib
import itertools
import pickle
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config.constants import EPSILON, FITNESS_PENALTY

if TYPE_CHECKING:
    from ..config.schemas import TournamentConfig
    from ..data_model.event import EventFactory, EventProperties
    from ..data_model.time import TimeSlot

logger = getLogger(__name__)


@dataclass(slots=True)
class FitnessBenchmark:
    """Benchmark for evaluating fitness scores."""

    config: TournamentConfig
    event_factory: EventFactory
    event_properties: EventProperties

    penalty: float = FITNESS_PENALTY
    cache_dir: Path = None
    timeslots: dict[frozenset[int], float] = field(default_factory=dict, init=False, repr=False)
    locations: np.ndarray = None
    opponents: np.ndarray = None
    flush_benchmarks: bool = False
    best_timeslot_score: float = None

    def __post_init__(self) -> None:
        """Post-initialization to validate run benchmark."""
        self.cache_dir = Path(".benchmarks_cache/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._run_benchmarks()

    def _run_benchmarks(self) -> None:
        """Load benchmarks from cache or run calculations if cache is invalid/missing."""
        config_hash = self._get_config_hash()
        cache_file = self.cache_dir / f"benchmark_cache_{config_hash}.pkl"
        if not self.flush_benchmarks:
            try:
                if cache_file.exists():
                    logger.debug("Loading fitness benchmarks from cache: %s", cache_file)
                    self._load_from_cache(cache_file)
                    return
            except (OSError, pickle.UnpicklingError, EOFError):
                logger.warning("Could not load cache file. Recalculating benchmarks.")
                cache_file.unlink(missing_ok=True)

        logger.info("No valid cache found. Calculating and caching new fitness benchmarks...")
        self._run_location_and_opponent_benchmarks()
        self._run_timeslot_benchmarks()
        self._save_to_cache(cache_file)

    def _load_from_cache(self, path: Path) -> None:
        """Load benchmark data from a pickle file."""
        try:
            with path.open("rb") as f:
                cached_data = pickle.load(f)
                self.timeslots = cached_data["timeslots"]
                self.locations = cached_data["locations"]
                self.opponents = cached_data["opponents"]
                self.best_timeslot_score = cached_data["best_timeslot_score"]
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.exception("Failed to load fitness benchmarks from cache.")

    def _save_to_cache(self, path: Path) -> None:
        """Save benchmark data to a pickle file."""
        data_to_cache = {
            "timeslots": self.timeslots,
            "locations": self.locations,
            "opponents": self.opponents,
            "best_timeslot_score": self.best_timeslot_score,
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("wb") as f:
                pickle.dump(data_to_cache, f)
            logger.info("Fitness benchmarks saved to cache: %s", path)
        except (OSError, pickle.PicklingError, EOFError):
            logger.exception("Failed to save fitness benchmarks to cache.")

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

    def _run_location_and_opponent_benchmarks(self) -> None:
        """Run the location consistency and opponent variety fitness benchmarking."""
        logger.info("Running location consistency and opponent variety benchmarks...")
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

        raw_scores = list(cache_scorer.values())
        logger.debug("Raw location/opponent scores: %s", raw_scores)
        locations = [abs((s - minimum_score) / diff) if s != 0 else 0 for s in raw_scores]
        opponents = [abs((s - maximum_score) / diff) if s != 0 else 0 for s in raw_scores]
        self.locations = np.array(locations, dtype=float)
        self.opponents = np.array(opponents, dtype=float)

        logger.debug("Location consistency scores:")
        for k, v in enumerate(self.locations):
            logger.debug("  %d location(s): %.6f", k, v)

        logger.debug("Opponent variety scores:")
        for k, v in enumerate(self.opponents):
            logger.debug("  %d opponent(s): %.6f", k, v)

        if not self.locations.any() or not self.opponents.any():
            logger.warning("No valid schedules could be generated.")
            return

    def _run_timeslot_benchmarks(self) -> None:
        """Run the time slot fitness benchmarking."""
        logger.info("Running break time consistency benchmarks...")
        logger.debug("Finding timeslots per round type:")
        timeslots_by_round = {r.roundtype: [ts.idx for ts in r.timeslots] for r in self.config.rounds}
        for rt, timeslot_idx in timeslots_by_round.items():
            roundtype = f"{rt:<10}"
            n_timeslots = len(timeslot_idx)
            ts_idx_to_str = ", ".join(str(ts) for ts in timeslot_idx)
            logger.debug("  %s: %d unique timeslots\n    %s", roundtype, n_timeslots, ts_idx_to_str)

        # Generate intra-round combinations
        logger.debug("Generating all possible schedules per round type:")
        round_slot_combos = {}
        for rt, num_needed in self.config.roundreqs.items():
            available_slots = timeslots_by_round.get(rt, [])
            round_slot_combos[rt] = list(itertools.combinations(available_slots, num_needed))
            logger.debug("  %s: %d round combinations", f"{rt:<10}", len(round_slot_combos[rt]))
            logger.debug("    %s", ", ".join(str(combo) for combo in round_slot_combos[rt]))

        # Filter, score, and store valid schedules
        logger.debug("Generating and filtering all possible team schedules")
        timeslot_objs = np.array(self.config.all_timeslots, dtype=object)
        valid_scored_schedules = []
        total_combinations = 0
        self.timeslots = {}
        for schedule_tuple in itertools.product(*round_slot_combos.values()):  # Cartesian product
            total_combinations += 1
            curr_indices = list(itertools.chain.from_iterable(schedule_tuple))
            curr_indices = np.array(curr_indices, dtype=int)
            curr_timeslots = timeslot_objs[curr_indices]
            if self._has_overlaps(curr_timeslots):
                continue

            score = self._score_break_time(curr_timeslots)
            valid_scored_schedules.append([score, curr_indices])
            self.timeslots[frozenset(curr_indices)] = score

        logger.debug("  Total combinations: %d", total_combinations)
        logger.debug("  Valid (non-overlapping): %d", len(self.timeslots))

        if not self.timeslots:
            logger.warning("No valid schedules could be generated.")
            return

        # Report results
        valid_scored_schedules.sort(key=lambda x: x[0], reverse=True)

        # Normalize scores for better comparison
        self.best_timeslot_score = valid_scored_schedules[0][0]
        if self.best_timeslot_score == 0:
            self.best_timeslot_score = 1  # Avoid division by zero
        logger.debug("Best timeslot score: %f", self.best_timeslot_score)

        for i, (score, indices) in enumerate(valid_scored_schedules):
            valid_scored_schedules[i][0] = score / self.best_timeslot_score
            self.timeslots[frozenset(indices)] /= self.best_timeslot_score

        unique_scores = Counter(score for score, _ in valid_scored_schedules)
        logger.debug("Unique scores found: %d", len(unique_scores))
        most_common_above_threshold = [(score, count) for score, count in unique_scores.items() if score > 0.10]
        most_common_above_threshold.sort(key=lambda x: (x[1], x[0]), reverse=True)

        total = 0
        for score, count in most_common_above_threshold[:20]:
            total += score
            logger.debug("  Score %f: %d occurrences", score, count)
        avg_score = total / 20
        logger.debug("Average score of top 20: %f", avg_score)

    def _has_overlaps(self, timeslots: list[TimeSlot]) -> bool:
        """Check if any timeslots in a sorted list overlap."""
        return any(timeslots[i + 1].overlaps(timeslots[i]) for i in range(len(timeslots) - 1))

    def _score_break_time(self, timeslots: list[TimeSlot]) -> float:
        """Calculate a break time fitness score for a non-overlapping combination of timeslots.

        A higher score is better.
        The score rewards longer, more consistent break times.

        """
        if len(timeslots) < 2:
            return 1.0  # Perfect score if only one event or no events

        starts = np.array([int(ts.start.timestamp()) for ts in timeslots], dtype=int)
        stops = np.array([int(ts.stop.timestamp()) for ts in timeslots], dtype=int)

        order = np.argsort(starts)
        starts_sorted = np.take_along_axis(starts, order, axis=0)
        stops_sorted = np.take_along_axis(stops, order, axis=0)

        breaks_seconds = starts_sorted[1:] - stops_sorted[:-1]
        breaks_minutes = breaks_seconds / 60
        if np.any(breaks_minutes < 0):
            return 0

        mean_break = np.mean(breaks_minutes, axis=0)
        if mean_break == 0:
            return 0

        std_dev = np.std(breaks_minutes, axis=0)

        coeff = std_dev / mean_break
        ratio = 1 / (1 + coeff)

        zeros = np.sum(breaks_minutes == 0)
        zeros_penalty = self.penalty**zeros

        return ratio * zeros_penalty
