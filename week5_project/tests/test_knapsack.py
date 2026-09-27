"""
test_knapsack.py
-------------------
pytest tests for src/dp/knapsack.py (Week 5, Part 2): the 0/1 knapsack problem.

Expected interface (document this in knapsack.py's own docstring when you write it):
    knapsack_recursive(weights: list[int], values: list[int], capacity: int) -> int
    knapsack_memo(weights, values, capacity) -> int
    knapsack_tabulation(weights, values, capacity) -> int
    trace_solution(weights, values, capacity) -> tuple[int, list[int]]
        -- (max achievable value, sorted list of indices of the items selected to
        achieve it). More than one valid selection can exist when items tie on
        value/weight ratio, so these tests check that the RETURNED selection is valid
        and optimal, not that it matches one specific "textbook" answer.

Only knapsack_recursive is exercised with more than a handful of items -- it's O(2^n),
so its test inputs stay small, mirroring how fibonacci_recursive is only tested up to
n=20 in test_fibonacci.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp.knapsack import (
    knapsack_recursive,
    knapsack_memo,
    knapsack_tabulation,
    trace_solution,
)


# Two classic worked examples with known optimal values (verified independently with
# a reference DP implementation, not hand-derived), used across every test class below
# so the three algorithms and trace_solution are all checked against the same truth.
EXAMPLE_1 = {"weights": [1, 3, 4, 5], "values": [1, 4, 5, 7], "capacity": 7, "expected": 9}
EXAMPLE_2 = {"weights": [10, 20, 30], "values": [60, 100, 120], "capacity": 50, "expected": 220}
EXAMPLES = [EXAMPLE_1, EXAMPLE_2]


class TestKnapsackRecursive:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert knapsack_recursive(case["weights"], case["values"], case["capacity"]) == case["expected"]

    def test_zero_capacity_is_worthless(self):
        assert knapsack_recursive([1, 2, 3], [10, 20, 30], 0) == 0

    def test_no_items_is_worthless(self):
        assert knapsack_recursive([], [], 10) == 0

    def test_single_item_that_fits(self):
        assert knapsack_recursive([5], [42], 10) == 42

    def test_single_item_that_does_not_fit(self):
        assert knapsack_recursive([15], [42], 10) == 0


class TestKnapsackMemo:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert knapsack_memo(case["weights"], case["values"], case["capacity"]) == case["expected"]

    def test_zero_capacity_is_worthless(self):
        assert knapsack_memo([1, 2, 3], [10, 20, 30], 0) == 0

    def test_handles_more_items_than_recursive_would_survive(self):
        # 40 items is roughly 2^40 branches for the naive version -- fine for memo/
        # tabulation's O(n*W), not something this suite asks knapsack_recursive to
        # attempt. Every weight is 1 and capacity is 20, so the answer is simply the
        # sum of the 20 largest values (21 through 40).
        weights = [1] * 40
        values = list(range(1, 41))
        assert knapsack_memo(weights, values, 20) == sum(range(21, 41))


class TestKnapsackTabulation:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert knapsack_tabulation(case["weights"], case["values"], case["capacity"]) == case["expected"]

    def test_zero_capacity_is_worthless(self):
        assert knapsack_tabulation([1, 2, 3], [10, 20, 30], 0) == 0

    def test_handles_more_items_than_recursive_would_survive(self):
        weights = [1] * 40
        values = list(range(1, 41))
        assert knapsack_tabulation(weights, values, 20) == sum(range(21, 41))


class TestAllImplementationsAgree:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_memo_matches_recursive(self, case):
        weights, values, capacity = case["weights"], case["values"], case["capacity"]
        assert knapsack_memo(weights, values, capacity) == knapsack_recursive(weights, values, capacity)

    @pytest.mark.parametrize("case", EXAMPLES)
    def test_tabulation_matches_recursive(self, case):
        weights, values, capacity = case["weights"], case["values"], case["capacity"]
        assert knapsack_tabulation(weights, values, capacity) == knapsack_recursive(weights, values, capacity)


class TestTraceSolution:
    """trace_solution has to return a selection that's both FEASIBLE (fits the
    capacity) and OPTIMAL (its value matches the known best) -- not just a plausible-
    looking list of indices."""

    @pytest.mark.parametrize("case", EXAMPLES)
    def test_selection_is_feasible_and_optimal(self, case):
        weights, values, capacity = case["weights"], case["values"], case["capacity"]
        max_value, selected = trace_solution(weights, values, capacity)

        assert max_value == case["expected"]

        # Every index must be a real, in-range item, used at most once (0/1 knapsack,
        # not unbounded knapsack).
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
