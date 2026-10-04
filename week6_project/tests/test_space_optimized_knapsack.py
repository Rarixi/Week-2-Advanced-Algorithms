"""
test_space_optimized_knapsack.py
-----------------------------------
pytest tests for src/dp_advanced/space_optimized_knapsack.py: the 0/1 knapsack problem,
full 2D table vs. space-optimized 1D array.

Uses the same two worked examples as Week 5's test_knapsack.py (known optimal values,
verified independently with a reference DP implementation), so this file is checking
the SAME correctness guarantee Week 5 established, plus the new claim this week adds:
that knapsack_1d produces identical answers to knapsack_2d while using less memory.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp_advanced.space_optimized_knapsack import knapsack_2d, knapsack_1d, trace_solution

EXAMPLE_1 = {"weights": [1, 3, 4, 5], "values": [1, 4, 5, 7], "capacity": 7, "expected": 9}
EXAMPLE_2 = {"weights": [10, 20, 30], "values": [60, 100, 120], "capacity": 50, "expected": 220}
EXAMPLES = [EXAMPLE_1, EXAMPLE_2]


class TestKnapsack2D:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert knapsack_2d(case["weights"], case["values"], case["capacity"]) == case["expected"]

    def test_zero_capacity_is_worthless(self):
        assert knapsack_2d([1, 2, 3], [10, 20, 30], 0) == 0

    def test_no_items_is_worthless(self):
        assert knapsack_2d([], [], 10) == 0

    def test_single_item_that_fits(self):
        assert knapsack_2d([5], [42], 10) == 42

    def test_single_item_that_does_not_fit(self):
        assert knapsack_2d([15], [42], 10) == 0


class TestKnapsack1D:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert knapsack_1d(case["weights"], case["values"], case["capacity"]) == case["expected"]

    def test_zero_capacity_is_worthless(self):
        assert knapsack_1d([1, 2, 3], [10, 20, 30], 0) == 0

    def test_no_items_is_worthless(self):
        assert knapsack_1d([], [], 10) == 0

    def test_single_item_that_fits(self):
        assert knapsack_1d([5], [42], 10) == 42

    def test_single_item_that_does_not_fit(self):
        assert knapsack_1d([15], [42], 10) == 0

    def test_does_not_reuse_an_item(self):
        # The classic bug this file's docstring warns about: looping the inner
        # capacity loop UPWARD instead of downward would silently turn this into
        # unbounded knapsack (each item usable any number of times). With one item
        # of weight 3 / value 10 and capacity 9, unbounded knapsack would answer 30
        # (three copies); 0/1 knapsack must answer 10 (one copy only).
        assert knapsack_1d([3], [10], 9) == 10

    def test_handles_more_items_than_recursive_would_survive(self):
        weights = [1] * 40
        values = list(range(1, 41))
        assert knapsack_1d(weights, values, 20) == sum(range(21, 41))


class TestBothVariantsAgree:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_1d_matches_2d(self, case):
        weights, values, capacity = case["weights"], case["values"], case["capacity"]
        assert knapsack_1d(weights, values, capacity) == knapsack_2d(weights, values, capacity)

    def test_agree_across_random_instances(self):
        import random
        rng = random.Random(42)
        for _ in range(30):
            n = rng.randint(0, 15)
            weights = [rng.randint(1, 20) for _ in range(n)]
            values = [rng.randint(1, 50) for _ in range(n)]
            capacity = rng.randint(0, 50)
            assert knapsack_1d(weights, values, capacity) == knapsack_2d(weights, values, capacity)


class TestTraceSolution:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_selection_is_feasible_and_optimal(self, case):
        weights, values, capacity = case["weights"], case["values"], case["capacity"]
        max_value, selected = trace_solution(weights, values, capacity)

        assert max_value == case["expected"]
        assert len(selected) == len(set(selected))
        assert all(0 <= i < len(weights) for i in selected)

        total_weight = sum(weights[i] for i in selected)
        total_value = sum(values[i] for i in selected)
        assert total_weight <= capacity
        assert total_value == max_value

    def test_empty_selection_for_zero_capacity(self):
        max_value, selected = trace_solution([1, 2, 3], [10, 20, 30], 0)
        assert max_value == 0
        assert selected == []
