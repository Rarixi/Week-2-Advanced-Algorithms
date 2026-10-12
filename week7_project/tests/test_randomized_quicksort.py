"""
test_randomized_quicksort.py
-------------------------------
Correctness tests for Part 1: randomized_quicksort.py.
"""

import random

from src.randomized.randomized_quicksort import (
    randomized_quicksort,
    deterministic_quicksort,
    randomized_quicksort_with_comparisons,
    deterministic_quicksort_with_comparisons,
)


class TestRandomizedQuicksortCorrectness:
    def test_empty_array(self):
        assert randomized_quicksort([]) == []

    def test_single_element(self):
        assert randomized_quicksort([5]) == [5]

    def test_already_sorted(self):
        arr = list(range(50))
        assert randomized_quicksort(arr) == arr

    def test_reverse_sorted(self):
        arr = list(range(50, 0, -1))
        assert randomized_quicksort(arr) == sorted(arr)

    def test_all_duplicates(self):
        arr = [7] * 30
        assert randomized_quicksort(arr) == arr

    def test_negative_numbers(self):
        arr = [-5, 3, -1, 0, 8, -9, 2]
        assert randomized_quicksort(arr) == sorted(arr)

    def test_does_not_mutate_input(self):
        arr = [3, 1, 2]
        original = arr.copy()
        randomized_quicksort(arr)
        assert arr == original

    def test_random_arrays_agree_with_builtin_sort(self):
        rng = random.Random(7)
        for _ in range(20):
            arr = [rng.randint(-1000, 1000) for _ in range(rng.randint(1, 200))]
            assert randomized_quicksort(arr) == sorted(arr)


class TestDeterministicQuicksortCorrectness:
    def test_empty_array(self):
        assert deterministic_quicksort([]) == []

    def test_already_sorted(self):
        arr = list(range(50))
        assert deterministic_quicksort(arr) == arr

    def test_reverse_sorted_worst_case_input(self):
        # This is deterministic_quicksort's O(n^2) worst case input; correctness
        # should still hold even though it will be slow.
        arr = list(range(500, 0, -1))
        assert deterministic_quicksort(arr) == sorted(arr)

    def test_random_arrays_agree_with_builtin_sort(self):
        rng = random.Random(8)
        for _ in range(20):
            arr = [rng.randint(-1000, 1000) for _ in range(rng.randint(1, 200))]
            assert deterministic_quicksort(arr) == sorted(arr)

    def test_does_not_mutate_input(self):
        arr = [3, 1, 2]
        original = arr.copy()
        deterministic_quicksort(arr)
        assert arr == original


class TestComparisonCounting:
    def test_with_comparisons_still_sorts_correctly(self):
        rng = random.Random(9)
        arr = [rng.randint(0, 500) for _ in range(300)]
        result, count = randomized_quicksort_with_comparisons(arr)
        assert result == sorted(arr)
        assert count > 0

    def test_deterministic_worst_case_has_far_more_comparisons(self):
        # On already-sorted input, deterministic_quicksort's last-element pivot
        # produces the maximally unbalanced partition every time -- its comparison
        # count should be close to n*(n-1)/2, while randomized stays close to the
        # much smaller 2n*ln(n) expectation.
        n = 1000
        sorted_input = list(range(n))
        _, det_count = deterministic_quicksort_with_comparisons(sorted_input)
        _, rand_count = randomized_quicksort_with_comparisons(sorted_input)
        assert det_count > rand_count * 5

    def test_deterministic_comparisons_match_worst_case_formula(self):
        n = 200
        sorted_input = list(range(n))
        _, det_count = deterministic_quicksort_with_comparisons(sorted_input)
        assert det_count == n * (n - 1) // 2
