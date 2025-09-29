"""Tests for genetic.fitness module."""

# from logging import CRITICAL, getLogger

# import pytest

# from fll_scheduler_ga.genetic import fitness as fitness_mod
# from fll_scheduler_ga.genetic.fitness import FitnessEvaluator, HardConstraintChecker, TwoTierCache

# # Silence logger output during tests
# getLogger(fitness_mod.__name__).setLevel(CRITICAL)


# class DummyTournamentConfig:
#     """A minimal TournamentConfig-like object for testing."""

#     def __init__(self, total_slots: int, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
#         """Initialize with total_slots and weights for (mean, var, range)."""
#         self.total_slots = total_slots
#         self.weights = weights


# class DummyBenchmark:
#     """Provides the dicts used as cold_cache for the evaluator."""

#     def __init__(self, timeslots: dict, locations: dict, opponents: dict) -> None:
#         """Initialize with dicts for each objective."""
#         # timeslots keys should be frozenset-like (as benchmark produces frozenset of TimeSlot)
#         self.timeslots = dict(timeslots)
#         self.locations = dict(locations)
#         self.opponents = dict(opponents)


# class DummyTeam:
#     """A minimal Team-like object for testing."""

#     def __init__(self, keys: tuple, rounds_needed_val: int = 0) -> None:
#         """Initialize with a tuple of keys and rounds_needed value."""
#         # keys: tuple of 3 keys (one per objective) matching evaluator ordering
#         self._keys = tuple(keys)
#         self._rounds_needed = rounds_needed_val
#         self.fitness = None

#     def get_fitness_keys(self) -> tuple:
#         """Return the keys used for fitness evaluation."""
#         return self._keys

#     def rounds_needed(self) -> int:
#         """Return the number of rounds needed for the team."""
#         return self._rounds_needed


# class DummySchedule:
#     """A minimal Schedule-like object for testing."""

#     def __init__(self, teams: list[DummyTeam], length: int | None = None) -> None:
#         """Initialize with a list of DummyTeam and optional length."""
#         self._teams = list(teams)
#         # if length not provided, len(schedule) returns number of teams by default
#         self._len = len(self._teams) if length is None else length
#         self.fitness = None

#     def all_teams(self) -> list[DummyTeam]:
#         """Return all teams in the schedule."""
#         return list(self._teams)

#     def __len__(self) -> int:
#         """Return the number of scheduled events."""
#         return self._len


# def test_two_tier_cache_get_behavior() -> None:
#     """Test TwoTierCache get method behavior for hits, misses, and key movement."""
#     hot = {"a": 1.0}
#     cold = {"b": 2.0}
#     cache = TwoTierCache(hot_cache=dict(hot), cold_cache=dict(cold))

#     # get hot key
#     val_a = cache.get("a")
#     assert val_a == 1.0
#     assert cache.hits == 1
#     assert cache.misses == 0

#     # get cold key moves it to hot and increments misses
#     val_b = cache.get("b")
#     assert val_b == 2.0
#     assert cache.misses == 1
#     assert "b" in cache.hot_cache
#     assert "b" not in cache.cold_cache

#     # subsequent get for b is a hit
#     val_b2 = cache.get("b")
#     assert val_b2 == 2.0
#     assert cache.hits == 2  # one for 'a', one for this
#     # missing key returns None
#     assert cache.get("nope") is None


# def test_hard_constraint_checker() -> None:
#     """Test HardConstraintChecker with various schedule states."""
#     # config expecting 3 total slots
#     cfg = DummyTournamentConfig(total_slots=3)
#     checker = HardConstraintChecker(cfg)

#     # empty schedule: should be falsy
#     empty_schedule = None
#     assert checker.check(empty_schedule) is False

#     # schedule with wrong length
#     teams = [DummyTeam((frozenset(), 1, 1)), DummyTeam((frozenset(), 1, 1))]
#     sched_wrong_len = DummySchedule(teams, length=2)
#     assert checker.check(sched_wrong_len) is False

#     # schedule with correct length but a team still needs rounds -> fail
#     teams = [
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=1),
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=0),
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=0),
#     ]
#     sched_needs_rounds = DummySchedule(teams, length=3)
#     assert checker.check(sched_needs_rounds) is False

#     # schedule with correct length and no rounds needed -> pass
#     teams_ok = [
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=0),
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=0),
#         DummyTeam((frozenset(), 1, 1), rounds_needed_val=0),
#     ]
#     sched_ok = DummySchedule(teams_ok, length=3)
#     assert checker.check(sched_ok) is True


# def test_fitness_evaluator_aggregate_and_evaluate_with_benchmark_shape() -> None:
#     """Test FitnessEvaluator methods with a benchmark and data shapes as expected."""
#     # Build "timeslot" sentinel objects to mimic TimeSlot instances used by benchmark
#     ts_a1 = object()
#     ts_a2 = object()
#     ts_b1 = object()
#     ts_b2 = object()

#     # keys are frozensets of timeslots (as benchmark stores)
#     timeslot_key1 = frozenset({ts_a1, ts_a2})
#     timeslot_key2 = frozenset({ts_b1, ts_b2})

#     # Create benchmark dictionaries with the proper key shapes
#     timeslots = {
#         timeslot_key1: 0.5,
#         timeslot_key2: 1.0,
#     }
#     # locations and opponents use integer keys (number of distinct locations/opponents)
#     locations = {1: 1.0, 2: 0.6}
#     opponents = {1: 0.2, 2: 0.8}

#     benchmark = DummyBenchmark(timeslots=timeslots, locations=locations, opponents=opponents)

#     # config: total_slots arbitrary, weights for mean,var,range
#     weights = (0.7, 0.2, 0.1)
#     cfg = DummyTournamentConfig(total_slots=2, weights=weights)

#     # create evaluator (this runs __post_init__ and populates cache_map)
#     evaluator = FitnessEvaluator(cfg, benchmark)

#     # Two teams: each must return keys in the order (timeslot_key, location_key, opponent_key)
#     team1 = DummyTeam((timeslot_key1, 1, 1))
#     team2 = DummyTeam((timeslot_key2, 2, 2))
#     schedule = DummySchedule([team1, team2], length=2)

#     # aggregate_team_fitnesses should set team.fitness and return scores per objective
#     scores = evaluator.aggregate_team_fitnesses(schedule.all_teams())
#     # scores is a tuple of 3 lists (one per objective)
#     assert len(scores) == 3
#     assert scores[0] == [timeslots[timeslot_key1], timeslots[timeslot_key2]]
#     assert scores[1] == [locations[1], locations[2]]
#     assert scores[2] == [opponents[1], opponents[2]]
#     # teams should have their .fitness set to tuple of 3 values
#     assert team1.fitness == (timeslots[timeslot_key1], locations[1], opponents[1])
#     assert team2.fitness == (timeslots[timeslot_key2], locations[2], opponents[2])

#     # mean scores
#     mean_s = evaluator.get_mean_scores(scores)
#     assert pytest.approx(mean_s[0]) == (timeslots[timeslot_key1] + timeslots[timeslot_key2]) / 2
#     assert pytest.approx(mean_s[1]) == (locations[1] + locations[2]) / 2
#     assert pytest.approx(mean_s[2]) == (opponents[1] + opponents[2]) / 2

#     # variation scores: convert generator to tuple
#     vari_s = tuple(evaluator.get_variation_scores(scores, mean_s))
#     # each coefficient should be between 0 and 1 (inclusive), and nonzero here
#     assert all(0 < v <= 1 for v in vari_s)

#     # range scores
#     rnge_s = tuple(evaluator.get_range_scores(scores))
#     # check expected numeric computation for first objective:
#     max_min_diff = max(scores[0]) - min(scores[0])
#     expected_range0 = 1 / (1 + max_min_diff)
#     assert pytest.approx(rnge_s[0]) == expected_range0

#     # now run evaluate to compute schedule.fitness
#     evaluator.evaluate(schedule)
#     assert isinstance(schedule.fitness, tuple)
#     assert len(schedule.fitness) == 3

#     # compute expected combined values using the evaluator helpers (mirrors evaluate)
#     mw, vw, rw = cfg.weights
#     expected = tuple((m * mw) + (v * vw) + (r * rw) for m, v, r in zip(mean_s, vari_s, rnge_s, strict=False))
#     # compare element-wise with approx
#     for got, exp in zip(schedule.fitness, expected, strict=False):
#         assert pytest.approx(got, rel=1e-9) == exp

#     # and cache info should be present for each objective
#     cache_info = evaluator.get_cache_info()
#     assert set(cache_info.keys()) == set(evaluator.cache_map.keys())
#     for info in cache_info.values():
#         assert "Hits" in info
#         assert "Misses" in info
