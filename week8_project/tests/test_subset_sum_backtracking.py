"""
test_subset_sum_backtracking.py
----------------------------------
Tests for the exact backtracking subset sum solver and its greedy heuristic
approximation. Checks correctness on known instances (including the known
case where greedy fails but backtracking succeeds), and that backtracking's
node count actually grows the way an exponential search should.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.complexity.subset_sum_backtracking import (
    subset_sum_backtracking,
    subset_sum_greedy_heuristic,
    is_valid_subset_sum,
)


class TestBacktrackingCorrectness:
    def test_finds_a_valid_subset_when_one_exists(self):
        numbers = [3, 34, 4, 12, 5, 2]
        target = 9
        found, subset = subset_sum_backtracking(numbers, target)
        assert found is True
        assert is_valid_subset_sum(subset, numbers, target)

    def test_reports_no_solution_correctly(self):
        numbers = [2, 4, 6, 8]
        target = 5  # all even numbers, can never sum to an odd target
        found, subset = subset_sum_backtracking(numbers, target)
        assert found is False
        assert subset is None

    def test_target_zero_is_the_empty_subset(self):
        found, subset = subset_sum_backtracking([1, 2, 3], 0)
        assert found is True
        assert subset == []


class TestGreedyHeuristic:
    def test_finds_exact_match_when_greedy_order_happens_to_work(self):
        numbers = [1, 2, 3, 9]
        chosen, achieved_sum, exact = subset_sum_greedy_heuristic(numbers, 9)
        assert exact is True
        assert achieved_sum == 9

    def test_can_miss_a_solution_backtracking_would_find(self):
        # The classic case, greedy takes 5 first and then can't reach 7
        # with what's left, even though 4 + 3 = 7 exists.
        numbers = [5, 4, 3]
        target = 7
        _chosen, _achieved, greedy_exact = subset_sum_greedy_heuristic(numbers, target)
        backtracking_found, _subset = subset_sum_backtracking(numbers, target)
        assert greedy_exact is False
        assert backtracking_found is True


class TestExponentialGrowth:
    def test_node_count_grows_with_input_size(self):
        # The real worst case for this search is a target just barely out of
        # reach, one more than the total sum, so remaining never hits 0 and
        # never goes negative early either, nothing to prune on, forcing the
        # full 2^n include/exclude tree to be explored every time.
        import random

        rng = random.Random(1)
        small_numbers = [rng.randint(100, 200) for _ in range(10)]
        large_numbers = small_numbers + [rng.randint(100, 200) for _ in range(6)]

        small_target = sum(small_numbers) + 1
        large_target = sum(large_numbers) + 1

        _found_small, _subset_small, stats_small = subset_sum_backtracking(
            small_numbers, small_target, collect_stats=True
        )
        _found_large, _subset_large, stats_large = subset_sum_backtracking(
            large_numbers, large_target, collect_stats=True
        )
        assert stats_large["nodes_visited"] > stats_small["nodes_visited"] * 10
