"""Time slot fitness benchmarking.

This module generates all possible valid time slot combinations for a single team
based on the tournament configuration and calculates a "break time fitness" score
for each combination. This helps identify the theoretically best and worst
schedules a team could receive, independent of other teams.
"""

import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations, product
from math import sqrt
from pathlib import Path

from ..data_model.event import EventFactory
from ..data_model.time import TimeSlot
from .config import TournamentConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitnessBenchmark:
    """Benchmark for evaluating fitness scores."""

    config: TournamentConfig
    event_factory: EventFactory
    cache_dir: Path = field(init=False, repr=False)
    penalty: float = 0.5  # Penalty value for penalizing worse scores
    timeslots: dict = field(default_factory=dict, init=False, repr=False)
    table: dict = field(default_factory=dict, init=False, repr=False)
    opponents: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to validate run benchmark."""
        self.cache_dir = Path(".benchmarks_cache/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_benchmarks()

    def run_benchmarks(self) -> None:
        """Load benchmarks from cache or run calculations if cache is invalid/missing."""
        config_hash = self._get_config_hash()
        cache_file = self.cache_dir / f"benchmark_cache_{config_hash}.pkl"

        try:
            if cache_file.exists():
                logger.info("Loading fitness benchmarks from cache: %s", cache_file)
                self._load_from_cache(cache_file)
                return
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.warning("Could not load cache file. Recalculating benchmarks.")
            cache_file.unlink(missing_ok=True)

        logger.info("No valid cache found. Calculating and caching new fitness benchmarks...")
        self.run_table_and_opponent_benchmarks()
        self.run_timeslot_benchmarks()
        self._save_to_cache(cache_file)

    def _load_from_cache(self, path: Path) -> None:
        """Load benchmark data from a pickle file."""
        with path.open("rb") as f:
            cached_data = pickle.load(f)
            self.timeslots = cached_data["timeslots"]
            self.table = cached_data["table"]
            self.opponents = cached_data["opponents"]

    def _save_to_cache(self, path: Path) -> None:
        """Save benchmark data to a pickle file."""
        data_to_cache = {
            "timeslots": self.timeslots,
            "table": self.table,
            "opponents": self.opponents,
        }
        # Ensure the cache directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(data_to_cache, f)
        logger.info("Fitness benchmarks saved to cache: %s", path)

    def _get_config_hash(self) -> int:
        """Generate a stable hash for the parts of the config that define the benchmark."""
        # Canonical representation of rounds
        round_tuples = tuple(
            (
                r.round_type,
                r.rounds_per_team,
                r.teams_per_round,
                r.start_time,
                r.stop_time,
                r.duration_minutes,
                r.num_locations,
            )
            for r in sorted(self.config.rounds, key=lambda x: x.start_time)
        )

        # Canonical representation of requirements
        req_tuple = tuple(sorted(self.config.round_requirements.items()))

        # Include the penalty in the hash
        config_representation = (round_tuples, req_tuple, self.penalty)
        return int(sha256(str(config_representation).encode()).hexdigest(), 16)

    def run_table_and_opponent_benchmarks(self) -> None:
        """Run the table consistency fitness benchmarking."""
        logger.info("Running table consistency and opponent variety benchmarks...")

        config_map = {r.round_type: r.teams_per_round for r in self.config.rounds}
        logger.debug("Finding events per round type:")

        total_locations_possible = 0
        total_locations_required = 0
        for rt, el in self.event_factory.build().items():
            if config_map[rt] == 2:
                total_locations_possible = len(el)
                total_locations_required += self.config.round_requirements[rt]

        minimum_locations = 1 / total_locations_possible
        maximum_locations = total_locations_required / total_locations_possible
        diff = maximum_locations - minimum_locations

        cache_scorer = {}
        for i in range(1, total_locations_required + 1):
            ratio = i / total_locations_possible
            cache_scorer[i] = 1 / (1 + ratio)

        maximum_score = cache_scorer[1]
        minimum_score = cache_scorer[total_locations_required]
        diff = maximum_score - minimum_score

        for num_loc, raw_score in cache_scorer.items():
            norm_table_score = abs((raw_score - minimum_score) / diff) if diff else 1
            table_penalty = self.penalty ** (num_loc - 1)
            table_score = norm_table_score * table_penalty
            self.table[num_loc] = table_score

            norm_opponent_score = abs((raw_score - maximum_score) / diff) if diff else 1
            opponent_penalty = self.penalty ** (total_locations_required - num_loc)
            opponent_score = norm_opponent_score * opponent_penalty
            self.opponents[num_loc] = opponent_score

        for k, v in self.table.items():
            logger.debug("Table score for %d table(s): %.6f", k, v)

        for k, v in self.opponents.items():
            logger.debug("Opponent score for %d opponent(s): %.6f", k, v)

        if not self.table or not self.opponents:
            logger.warning("No valid schedules could be generated.")
            return

    def run_timeslot_benchmarks(self) -> None:
        """Run the time slot fitness benchmarking."""
        logger.info("Running break time consistency benchmarks...")
        logger.debug("Finding timeslots per round type:")
        timeslots_by_round = defaultdict(list)
        for rt, el in self.event_factory.build().items():
            timeslots_by_round[rt] = sorted({event.timeslot for event in el})
            logger.debug("  %s: %d unique timeslots", f"{rt:<10}", len(timeslots_by_round[rt]))

        # Generate intra-round combinations
        logger.debug("Generating all possible schedules per round type:")
        round_slot_combos = {}
        for rt, num_needed in self.config.round_requirements.items():
            available_slots = timeslots_by_round.get(rt, [])
            round_slot_combos[rt] = list(combinations(available_slots, num_needed))
            logger.debug("  %s: %d round combinations", f"{rt:<10}", len(round_slot_combos[rt]))

        valid_scored_schedules = []
        total_generated = 0
        logger.debug("Generating and filtering all possible team schedules")

        # Filter, score, and store valid schedules
        for schedule_tuple in product(*round_slot_combos.values()):  # Cartesian product
            total_generated += 1
            current_combination = [slot for combo in schedule_tuple for slot in combo]
            current_combination.sort(key=lambda ts: ts.start)

            if not FitnessBenchmark.has_overlaps(current_combination):
                score = FitnessBenchmark.score_break_time(current_combination, self.penalty)
                valid_scored_schedules.append([score, current_combination])
                self.timeslots[frozenset(current_combination)] = score

        logger.debug("  Total potential: %d", total_generated)
        logger.debug("  Valid (non-overlapping): %d", len(self.timeslots))

        if not self.timeslots:
            logger.warning("No valid schedules could be generated.")
            return

        # Report results
        valid_scored_schedules.sort(key=lambda x: x[0], reverse=True)

        # Normalize scores for better comparison
        if (best_score := valid_scored_schedules[0][0]) == 0:
            best_score = 1  # Avoid division by zero

        for scores in valid_scored_schedules:
            scores[0] /= best_score
            self.timeslots[frozenset(scores[1])] /= best_score

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

    @staticmethod
    def has_overlaps(timeslots: list[TimeSlot]) -> bool:
        """Check if any timeslots in a sorted list overlap."""
        return any(timeslots[i + 1].overlaps(timeslots[i]) for i in range(len(timeslots) - 1))

    @staticmethod
    def score_break_time(timeslots: list[TimeSlot], penalty: float) -> float:
        """Calculate a break time fitness score for a non-overlapping combination of timeslots.

        A higher score is better.
        The score rewards longer, more consistent break times.

        """
        if len(timeslots) < 2:
            return 1.0  # Perfect score if only one event or no events

        # Assumes timeslots are already sorted by start time
        breaks_in_minutes = []
        for i in range(1, len(timeslots)):
            break_duration = timeslots[i].start - timeslots[i - 1].stop
            breaks_in_minutes.append(break_duration.total_seconds() / 60)

        # Check for negative breaks, which indicates an overlap that slipped through
        if any(b < 0 for b in breaks_in_minutes):
            return 0.0

        if (n := len(breaks_in_minutes)) == 0:
            return 1.0

        if (mean_break := sum(breaks_in_minutes) / n) == 0:
            return 0.0

        # Calculate standard deviation to measure consistency
        sum_sq_diff = sum((b - mean_break) ** 2 for b in breaks_in_minutes)
        zeros = breaks_in_minutes.count(0)
        std_dev = sqrt(sum_sq_diff / n)

        # Use coefficient of variation (std_dev / mean) to normalize.
        # A lower coefficient is better (less variation relative to the average break).
        # The score is inverted so that a lower coefficient gives a higher score.
        coefficient = std_dev / mean_break
        ratio = 1 / (1 + coefficient)
        b_penalty = penalty**zeros
        return ratio * b_penalty
