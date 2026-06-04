"""Base class for fitness evaluators."""

import itertools
from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from fll_scheduler_ga.constants import EPSILON, FitnessObjective

if TYPE_CHECKING:
    from fll_scheduler_ga.domain.model import EventProperties, EventRepository, TournamentConfig


logger = getLogger(__name__)

_MAX_NP_INT: int = np.iinfo(np.int64).max
_EPSILON: float = EPSILON


@dataclass(slots=True)
class BreakTimeFitnessEvaluator:
    """Evaluator for break time fitness."""

    benchmark_best_timeslot_score: float
    minbreak_target: int
    minbreak_penalty: float
    zeros_penalty: float

    def __call__(self, start: np.ndarray, stop_active: np.ndarray, stop_cycle: np.ndarray) -> np.ndarray:
        """Vectorized break time scoring."""
        start_sorted, stop_active_sorted, stop_cycle_sorted = self._sort_evt_by_start(start, stop_active, stop_cycle)
        break_active, break_cycle = self._compute_break_dur(start_sorted, stop_active_sorted, stop_cycle_sorted)
        overlap_mask = (break_cycle < 0).any(axis=2)
        ratio, mean_zero_mask = self._ratio(break_cycle)
        score = ratio * self._zero_penalty(break_cycle) * self._minbreak_penalty(break_active)
        score[mean_zero_mask | overlap_mask] = 0.0
        return score / self.benchmark_best_timeslot_score

    def _sort_evt_by_start(
        self, start: np.ndarray, stop_active: np.ndarray, stop_cycle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = start.argsort(axis=2)
        return (
            np.take_along_axis(start, order, axis=2),
            np.take_along_axis(stop_active, order, axis=2),
            np.take_along_axis(stop_cycle, order, axis=2),
        )

    def _compute_break_dur(
        self, start: np.ndarray, stop_active: np.ndarray, stop_cycle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        start_next = start[:, :, 1:]
        break_active = np.subtract(start_next, stop_active[:, :, :-1]) / 60
        break_cycle = np.subtract(start_next, stop_cycle[:, :, :-1]) / 60
        return break_active, break_cycle

    def _ratio(self, break_cycle: np.ndarray) -> tuple[np.ndarray, ...]:
        valid_mask = break_cycle >= 0
        count = valid_mask.sum(axis=2, dtype=int)
        mean = break_cycle.sum(axis=2) / count
        mean_zero_mask = mean == 0
        mean[mean_zero_mask] = _EPSILON
        diff_sq = np.square(break_cycle - mean[:, :, np.newaxis])
        variance = diff_sq.sum(axis=2) / count
        std_dev = np.sqrt(variance)
        coeff = std_dev / mean
        ratio = 1 / (1 + coeff)
        return ratio, mean_zero_mask

    def _minbreak_penalty(self, break_active: np.ndarray) -> np.ndarray:
        count = (break_active < self.minbreak_target).sum(axis=2)
        short_mask = (break_active < self.minbreak_target) & (break_active > 0)
        if short_mask.any():
            max_diff = np.where(short_mask, self.minbreak_target - break_active, 0.0).max(axis=2)
            max_diff /= self.minbreak_target
        else:
            max_diff = np.zeros_like(count)
        return self.minbreak_penalty ** (count + max_diff)

    def _zero_penalty(self, break_cycle: np.ndarray) -> np.ndarray:
        return self.zeros_penalty ** (break_cycle == 0).sum(axis=2)


@dataclass(slots=True)
class LocationConsistencyFitnessEvaluator:
    """Location consistency fitness evaluator."""

    loc_weight_round_inter: float
    loc_weight_round_intra: float
    match_rt: np.ndarray
    rt_array: np.ndarray

    def __call__(self, loc_id: np.ndarray, roundtype_id: np.ndarray) -> np.ndarray:
        """Calculate location consistency score, prioritizing inter-round over intra-round consistency."""
        n_pop, n_teams, _ = loc_id.shape
        shape = (n_pop, n_teams)
        max_loc_idx = loc_id.max()
        if self.match_rt.size < 1 or max_loc_idx < 0:
            return np.ones(shape, dtype=float)
        rt_loc_count, match_rt_mask = self._build_rt_loc_count(loc_id, roundtype_id, max_loc_idx, n_pop, n_teams)
        participated_in_rt = rt_loc_count.sum(axis=3, dtype=int) > 0
        num_participated_rt = participated_in_rt.sum(axis=2, dtype=float)
        inter_score = self._inter_round_score(rt_loc_count, num_participated_rt, shape)
        intra_score = self._intra_round_score(rt_loc_count, participated_in_rt, num_participated_rt, shape)
        score = (inter_score * self.loc_weight_round_inter) + (intra_score * self.loc_weight_round_intra)
        score[match_rt_mask.sum(axis=2) <= 1] = 1.0
        return score

    def _build_rt_loc_count(
        self, loc_id: np.ndarray, roundtype_id: np.ndarray, max_loc_idx: int, n_pop: int, n_teams: int
    ) -> tuple[np.ndarray, np.ndarray]:
        max_rt_id = max(roundtype_id.max(), self.match_rt.max())
        is_match_rt_lookup = np.zeros(max_rt_id + 1, dtype=bool)
        is_match_rt_lookup[self.match_rt] = True
        match_rt_mask = is_match_rt_lookup[roundtype_id] & (loc_id >= 0)
        pop_idx, team_idx, _ = match_rt_mask.nonzero()
        rt_loc_count = np.zeros((n_pop, n_teams, self.match_rt.size, max_loc_idx + 1), dtype=int)
        rt_loc_count[pop_idx, team_idx, self.rt_array[roundtype_id[match_rt_mask]], loc_id[match_rt_mask]] = 1
        return rt_loc_count, match_rt_mask

    def _inter_round_score(
        self, rt_loc_count: np.ndarray, num_participated_rt: np.ndarray, shape: tuple[int, ...]
    ) -> np.ndarray:
        score = np.ones(shape, dtype=float)
        if self.match_rt.size >= 2:
            loc_used_per_rt = (rt_loc_count > 0).sum(axis=2, dtype=float)
            intersection = (loc_used_per_rt == num_participated_rt[:, :, np.newaxis]).sum(axis=2)
            union = (loc_used_per_rt > 0).sum(axis=2)
            valid = union > 0
            score[valid] = intersection[valid] / union[valid]
        return score

    def _intra_round_score(
        self,
        rt_loc_count: np.ndarray,
        participated_in_rt: np.ndarray,
        num_participated_rt: np.ndarray,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        unique_loc: np.ndarray = (rt_loc_count > 0).sum(axis=3, dtype=float)
        unique_loc[unique_loc == 0] = _EPSILON
        per_rt = 1.0 / unique_loc
        per_rt[~participated_in_rt] = 1.0
        valid = num_participated_rt > 0
        score = np.ones(shape, dtype=float)
        score[valid] = per_rt.sum(axis=2)[valid] / num_participated_rt[valid]
        return score


@dataclass(slots=True)
class OpponentVarietyFitnessEvaluator:
    """Opponent variety fitness evaluator."""

    benchmark_opponents: np.ndarray
    single_roundtypes: np.ndarray

    def __call__(self, paired_evt_id: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Vectorized opponent variety scoring."""
        valid_opp = paired_evt_id >= 0
        paired_evt_id[~valid_opp] = 0
        sched_idx = np.arange(arr.shape[0], dtype=int)[:, None, None]
        opponents = arr[sched_idx, paired_evt_id]
        opponents[~valid_opp] = _MAX_NP_INT
        opponents.sort(axis=2)
        valid_mask = opponents[:, :, :-1] >= 0
        unique_counts = ((np.diff(opponents, axis=2) != 0) & valid_mask).sum(axis=2)
        if self.single_roundtypes.size == 0:
            unique_counts = unique_counts + 1
        return self.benchmark_opponents[unique_counts]


@dataclass(slots=True)
class FitnessEvaluator:
    """Base class for fitness evaluators."""

    n_team: int
    n_max_evt_per_team: int
    evt_prop: EventProperties
    agg_weights: tuple[float, ...]
    min_fitness_weight: float
    obj_weights: np.ndarray
    breaktime_evaluator: BreakTimeFitnessEvaluator
    locationconsistency_evaluator: LocationConsistencyFitnessEvaluator
    opponentvariety_evaluator: OpponentVarietyFitnessEvaluator
    n_objectives: int = len(tuple(FitnessObjective))

    def evaluate(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an entire population of schedules.

        Args:
            arr (np.ndarray): Shape (pop_size, num_events). The core data.

        Returns:
            np.ndarray: Final fitness scores for the population. Shape (pop_size, num_objectives).
            np.ndarray: All team scores for the population. Shape (pop_size, num_teams, num_objectives).

        """
        t_evt = self.get_team_evt(arr)
        t_fit = np.zeros((arr.shape[0], self.n_team, self.n_objectives), dtype=float)
        t_fit[:, :, 0] = self.breaktime_evaluator(
            self.evt_prop.start[t_evt], self.evt_prop.stop_active[t_evt], self.evt_prop.stop_cycle[t_evt]
        )
        t_fit[:, :, 1] = self.locationconsistency_evaluator(
            self.evt_prop.loc_idx[t_evt], self.evt_prop.roundtype_idx[t_evt]
        )
        t_fit[:, :, 2] = self.opponentvariety_evaluator(self.evt_prop.paired_idx[t_evt], arr)
        return self.aggregate(t_fit), t_fit

    def get_team_evt(self, arr: np.ndarray) -> np.ndarray:
        """Invert the (event -> team) mapping to a (team -> events) mapping for the entire population."""
        team_evt = np.full((arr.shape[0], self.n_team, self.n_max_evt_per_team), -1, dtype=int)
        sched_idx, event_idx = (arr >= 0).nonzero()
        if sched_idx.size == 0:
            return team_evt
        key = (sched_idx * self.n_team) + arr[sched_idx, event_idx]
        count = np.bincount(key, minlength=arr.shape[0] * self.n_team)
        order = key.argsort()
        sorted_event_idx = event_idx[order]
        group_start = np.zeros_like(count, dtype=int)
        group_start[1:] = count[:-1].cumsum()
        within_group_idx = np.arange(sorted_event_idx.size, dtype=int) - group_start.repeat(count)
        sortkey = key[order]
        pop_idx_sorted = sortkey // self.n_team
        team_idx_sorted = sortkey % self.n_team
        mask = within_group_idx < self.n_max_evt_per_team
        team_evt[pop_idx_sorted[mask], team_idx_sorted[mask], within_group_idx[mask]] = sorted_event_idx[mask]
        return team_evt

    def aggregate(self, team_fit: np.ndarray) -> np.ndarray:
        """Aggregate team fitness scores into schedule fitness scores."""
        min_s = team_fit.min(axis=1)
        mean_s = team_fit.mean(axis=1)
        mean_s = (mean_s * (1.0 - self.min_fitness_weight)) + (min_s * self.min_fitness_weight)
        mean_s[mean_s == 0] = _EPSILON
        coeff_s = team_fit.std(axis=1) / mean_s
        vari_s = 1.0 / (1.0 + coeff_s)
        ptp = team_fit.max(axis=1) - min_s
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
        max_possible, max_required, non_match = self._count_requirements()
        score = self._build_raw_scores(max_possible, max_required, non_match)
        logger.debug("Raw location/opponent scores: %s", score)
        opp = np.array(self._normalize_to_opponent_scores(score, max_required), dtype=float)
        for k, v in enumerate(opp):
            logger.debug("  %d opponent(s): %.6f", k, v)
        if not opp.any():
            logger.warning("No valid schedules could be generated.")
            return np.array([])
        return opp

    def _count_requirements(self) -> tuple[int, int, int]:
        """Tally match/non-match round requirements from config."""
        max_possible = max_required = non_match = 0
        for r in self.config.rounds:
            events = self.event_repo.roundtypes.get(r.roundtype_idx, [])
            tpr = self.config.round_idx_to_tpr[r.roundtype_idx]
            roundreq = self.config.roundreqs[r.roundtype]
            if tpr == 2:
                max_possible += len(events)
                max_required += roundreq
            elif tpr == 1:
                non_match += roundreq
        return max_possible, max_required, non_match

    def _build_raw_scores(self, max_possible: int, max_required: int, non_match: int) -> list[float]:
        """Build raw opponent-count → score lookup (index = opponent count)."""
        score: list[float] = [0.0] * (max_required + non_match + 1)
        for k in range(1, max_required + 1):
            score[k] = 1 / (1 + k / max_possible)
        if non_match > 0:
            score[max_required + non_match] = 0.0
        return score

    def _normalize_to_opponent_scores(self, score: list[float], max_required: int) -> list[float]:
        """Normalize raw scores to [0, 1] range relative to best/worst opponent counts."""
        maximum = score[1]
        minimum = score[max_required]
        diff = max(maximum - minimum, EPSILON)
        return [(maximum - s) / diff if s != 0 else 0.0 for s in score]


@dataclass(slots=True)
class FitnessBenchmarkBreaktime:
    """Benchmark for break time consistency fitness."""

    config: TournamentConfig
    minbreak_target: int
    minbreak_penalty: float
    zeros_penalty: float

    def benchmark(self) -> float:
        """Run the break time consistency fitness benchmarking."""
        logger.info("Running break time consistency benchmarks...")
        all_start, all_stop_active, all_stop_cycle = self._build_timeslot_array()
        logger.debug("Finding timeslots per round type:")
        ts_by_round = {r.roundtype: [ts.idx for ts in r.timeslots] for r in self.config.rounds}
        round_slot_combo = self._gen_intra_round_breaktime_combo(ts_by_round)
        logger.debug("Generating and filtering all possible team schedules")
        flattened = [list(itertools.chain.from_iterable(p)) for p in itertools.product(*round_slot_combo)]
        idx_matrix = np.array(flattened, dtype=int) if flattened else None
        if idx_matrix is None:
            logger.warning("No possible schedules could be generated.")
            return 0
        logger.debug("indices_matrix (Shape: %s):\n%s", idx_matrix.shape, idx_matrix)
        logger.debug("total_combinations: %d", idx_matrix.shape[0])
        logger.debug("calculating breaktime scores vectorized...")
        valid_score, _ = self.score_breaktime(idx_matrix, all_start, all_stop_active, all_stop_cycle)
        if valid_score.shape[0] == 0:
            logger.warning("No valid schedules could be generated.")
            return 0
        logger.debug("num_valid: %d", valid_score.shape[0])
        best_score = valid_score.max() or 1  # Avoid division by zero
        logger.debug("Best timeslot score: %f", best_score)
        self._log_score_distribution(valid_score, best_score)
        return best_score

    def _build_timeslot_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract start/stop timestamps from config timeslots as integer arrays."""
        return (
            np.array([int(ts.start.timestamp()) for ts in self.config.all_timeslots], dtype=int),
            np.array([int(ts.stop_active.timestamp()) for ts in self.config.all_timeslots], dtype=int),
            np.array([int(ts.stop_cycle.timestamp()) for ts in self.config.all_timeslots], dtype=int),
        )

    def _gen_intra_round_breaktime_combo(self, ts_by_round: dict[str, list[int]]) -> list[tuple[tuple[int, ...], ...]]:
        """Generate all intra-round breaktime combinations."""
        logger.debug("Generating all possible schedules per round type:")
        round_slot_combo = []
        for rt, num_needed in self.config.roundreqs.items():
            ts_idx = ts_by_round.get(rt, [])
            combo = tuple(itertools.combinations(ts_idx, num_needed))
            round_slot_combo.append(combo)
            logger.debug("  roundtype: %s", rt)
            logger.debug("    %d timeslots", len(ts_idx))
            logger.debug("      timeslots: %s", ts_idx)
            logger.debug("    %d combinations", len(combo))
            logger.debug("      combinations: %s", combo)
        return round_slot_combo

    def _log_score_distribution(self, valid_score: np.ndarray, best_score: float) -> None:
        """Log the distribution of scores."""
        unique_score = Counter(valid_score / best_score)
        logger.debug("Unique scores found: %d", len(unique_score))
        logger.debug("Number of best scores: %d", unique_score.get(1.0, 0))
        most_common = unique_score.most_common(50)
        for score, count in most_common:
            logger.debug("  Score %s: %d occurrences", f"{score:<.16f}", count)
        logger.debug("Average score of most common: %f", sum(s for s, _ in most_common) / len(most_common))

    def score_breaktime(
        self, idx: np.ndarray, start: np.ndarray, stop_active: np.ndarray, stop_cycle: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Calculate breaktime fitnesses vectorized."""
        start_ = start[idx]
        order = start_.argsort(axis=1)
        start_sorted = np.take_along_axis(start_, order, axis=1)
        stop_active_sorted = np.take_along_axis(stop_active[idx], order, axis=1)
        stop_cycle_sorted = np.take_along_axis(stop_cycle[idx], order, axis=1)
        start_next = start_sorted[:, 1:]
        break_active = (start_next - stop_active_sorted[:, :-1]) / 60
        break_cycle = (start_next - stop_cycle_sorted[:, :-1]) / 60
        overlap_mask = (break_cycle < 0).any(axis=1)
        ratio, mean_zero_mask = self._ratio(break_cycle)
        score = ratio * self._zero_penalty(break_cycle) * self._minbreak_penalty(break_active)
        score[mean_zero_mask | overlap_mask] = 0.0
        return score[~overlap_mask], idx[~overlap_mask]

    def _ratio(self, break_cycle: np.ndarray) -> tuple[np.ndarray, ...]:
        valid_mask = break_cycle >= 0
        count = valid_mask.sum(axis=1, dtype=int)
        mean = break_cycle.sum(axis=1) / count
        mean_zero_mask = mean == 0
        mean[mean_zero_mask] = _EPSILON
        diff_sq = np.square(break_cycle - mean[:, np.newaxis])
        variance = diff_sq.sum(axis=1) / count
        std_dev = np.sqrt(variance)
        coeff = std_dev / mean
        ratio = 1 / (1 + coeff)
        return ratio, mean_zero_mask

    def _minbreak_penalty(self, break_active: np.ndarray) -> np.ndarray:
        count = (break_active < self.minbreak_target).sum(axis=1)
        short_mask = (break_active < self.minbreak_target) & (break_active > 0)
        if short_mask.any():
            max_diff = np.where(short_mask, self.minbreak_target - break_active, 0.0).max(axis=1)
            max_diff /= self.minbreak_target
        else:
            max_diff = np.zeros_like(count)
        return self.minbreak_penalty ** (count + max_diff)

    def _zero_penalty(self, break_cycle: np.ndarray) -> np.ndarray:
        return self.zeros_penalty ** (break_cycle == 0).sum(axis=1)
