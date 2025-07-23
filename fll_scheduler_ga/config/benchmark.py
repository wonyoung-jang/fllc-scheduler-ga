"""Time slot fitness benchmarking.

This module generates all possible valid time slot combinations for a single team
based on the tournament configuration and calculates a "break time fitness" score
for each combination. This helps identify the theoretically best and worst
schedules a team could receive, independent of other teams.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from math import sqrt
from pathlib import Path

from ..data_model.event import EventFactory
from ..data_model.time import TimeSlot
from .config import TournamentConfig, load_tournament_config

logger = logging.getLogger(__name__)
PENALTY = 1e-16  # Penalty value for penalizing worse scores


@dataclass(slots=True)
class TableConsistencyBenchmark:
    """Benchmark for evaluating table consistency fitness scores."""

    config: TournamentConfig
    event_factory: EventFactory
    cache: dict = field(default_factory=dict, init=False, repr=False)

    def run(self) -> None:
        """Run the table consistency fitness benchmarking."""
        logger.info("Running table consistency fitness benchmarking...")
        self.cache = {
            "table": {},
            "opponents": {},
        }

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

        for k, v in cache_scorer.items():
            self.cache["table"][k] = abs((v - minimum_score) / diff) if diff else 1
            self.cache["opponents"][k] = abs((v - maximum_score) / diff) if diff else 1
            logger.debug("  %s: %.3f", f"{k:<10}", self.cache["table"][k])
            logger.debug("  %s: %.3f", f"{k:<10}", self.cache["opponents"][k])

        if not self.cache:
            logger.warning("No valid schedules could be generated.")
            return


@dataclass(slots=True)
class BreakTimeFitnessBenchmark:
    """Benchmark for evaluating break time fitness scores."""

    config: TournamentConfig
    event_factory: EventFactory
    cache: dict = field(default_factory=dict, init=False, repr=False)

    def run(self) -> None:
        """Run the time slot fitness benchmarking."""
        logger.info("Running time slot fitness benchmarking...")
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

            if not BreakTimeFitnessBenchmark.has_overlaps(current_combination):
                score = BreakTimeFitnessBenchmark.score_break_time(current_combination, PENALTY)
                valid_scored_schedules.append([score, current_combination])
                self.cache[frozenset(current_combination)] = score

        logger.debug("  Total potential: %d", total_generated)
        logger.debug("  Valid (non-overlapping): %d", len(self.cache))

        if not self.cache:
            logger.warning("No valid schedules could be generated.")
            return

        # Report results
        valid_scored_schedules.sort(key=lambda x: x[0], reverse=True)

        # Normalize scores for better comparison
        if (best_score := valid_scored_schedules[0][0]) == 0:
            best_score = 1  # Avoid division by zero

        for scores in valid_scored_schedules:
            scores[0] /= best_score
            self.cache[frozenset(scores[1])] /= best_score

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

        logger.debug("All unique scores:")
        for score, count in unique_scores.items():
            logger.debug("  Score %f: %d occurrences", score, count)

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


def main() -> None:
    """Run the time slot fitness benchmarking."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="time_fitness.log",
        filemode="w",
    )
    # --- 1. SETUP ---
    config, _ = load_tournament_config(Path("fll_scheduler_ga/config.ini"))
    event_factory = EventFactory(config)
    benchmark = BreakTimeFitnessBenchmark(config, event_factory)
    benchmark.run()
    table_benchmark = TableConsistencyBenchmark(config, event_factory)
    table_benchmark.run()


if __name__ == "__main__":
    main()
