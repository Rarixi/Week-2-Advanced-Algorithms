"""
test_randomized_selection.py
-------------------------------
Correctness tests for Part 2: randomized_selection.py.
"""

import random

from src.randomized.randomized_selection import (
    randomized_select,
    deterministic_select,
    randomized_select_with_stats,
)


class TestRandomizedSelect:
    def test_minimum(self, small_random_array):
        assert randomized_select(small_random_array, 1) == min(small_random_array)

    def test_maximum(self, small_random_array):
        n = len(small_random_array)
        assert randomized_select(small_random_array, n) == max(small_random_array)

    def test_median(self, small_random_array):
        s = sorted(small_random_array)
        k = (len(s) + 1) // 2
        assert randomized_select(small_random_array, k) == s[k - 1]

    def test_every_rank_matches_sorted_order(self):
        rng = random.Random(10)
        arr = [rng.randint(-500, 500) for _ in range(100)]
        s = sorted(arr)
        for k in range(1, len(arr) + 1):
            assert randomized_select(arr, k) == s[k - 1]

    def test_does_not_mutate_input(self, small_random_array):
        original = small_random_array.copy()
        randomized_select(small_random_array, 3)
        assert small_random_array == original

    def test_single_element(self):
        assert randomized_select([42], 1) == 42

    def test_duplicates(self):
        arr = [5, 5, 5, 1, 5]
        s = sorted(arr)
        for k in range(1, len(arr) + 1):
            assert randomized_select(arr, k) == s[k - 1]


class TestDeterministicSelect:
    def test_matches_sorted_order(self):
        rng = random.Random(11)
        arr = [rng.randint(-500, 500) for _ in range(100)]
        s = sorted(arr)
        for k in [1, 50, 100]:
            assert deterministic_select(arr, k) == s[k - 1]

    def test_does_not_mutate_input(self, small_random_array):
        original = small_random_array.copy()
        deterministic_select(small_random_array, 3)
        assert small_random_array == original

    def test_agrees_with_randomized_select(self):
        rng = random.Random(12)
        for _ in range(15):
            arr = [rng.randint(-1000, 1000) for _ in range(rng.randint(1, 150))]
            k = rng.randint(1, len(arr))
            assert deterministic_select(arr, k) == randomized_select(arr, k)


class TestSelectionStats:
    def test_stats_structure(self):
        rng = random.Random(13)
        arr = [rng.randint(0, 10000) for _ in range(500)]
        value, stats = randomized_select_with_stats(arr, 250)
        assert value == sorted(arr)[249]
        assert stats["recursion_depth"] >= 1
        assert stats["total_calls"] >= 1
        assert 0 <= stats["good_split_count"] <= stats["total_calls"]

    def test_recursion_depth_stays_small_on_average(self):
        # Randomized selection's expected recursion depth is O(log n); it should
        # stay nowhere near n itself even for a moderately large array.
        rng = random.Random(14)
        arr = list(range(5000))
        rng.shuffle(arr)
        _, stats = randomized_select_with_stats(arr, 2500)
        assert stats["recursion_depth"] < 100
