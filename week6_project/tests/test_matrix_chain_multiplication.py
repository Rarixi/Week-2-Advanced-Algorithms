"""
test_matrix_chain_multiplication.py
--------------------------------------
pytest tests for src/dp_advanced/matrix_chain_multiplication.py.

EXAMPLE_CLASSIC is the textbook MCM example (dims = [30,35,15,5,10,20,25], a 6-matrix
chain), whose optimal cost (15125) and one valid optimal parenthesization are widely
published, independently of this codebase -- used here as a known-correct reference
rather than a hand-derived or self-verified value.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.dp_advanced.matrix_chain_multiplication import (
    mcm_recursive,
    mcm_memo,
    mcm_tabulation,
    optimal_parenthesization,
)

EXAMPLE_CLASSIC = {"dims": [30, 35, 15, 5, 10, 20, 25], "expected": 15125}
EXAMPLE_SMALL = {"dims": [10, 20, 30], "expected": 6000}  # one split point, trivially checkable by hand
EXAMPLES = [EXAMPLE_CLASSIC, EXAMPLE_SMALL]


class TestMCMRecursive:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert mcm_recursive(case["dims"]) == case["expected"]

    def test_single_matrix_costs_nothing(self):
        # A "chain" of one matrix needs no multiplication at all.
        assert mcm_recursive([10, 20]) == 0

    def test_two_matrices_has_exactly_one_option(self):
        # p x q times q x r costs p*q*r; no split-point choice exists for only 2 matrices.
        assert mcm_recursive([4, 5, 6]) == 4 * 5 * 6


class TestMCMMemo:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert mcm_memo(case["dims"]) == case["expected"]

    def test_single_matrix_costs_nothing(self):
        assert mcm_memo([10, 20]) == 0

    def test_handles_a_longer_chain_than_recursive_would_survive(self):
        # 15 matrices is well past what naive recursion should be asked to attempt
        # (Catalan(14) = 2,674,440 distinct parenthesizations) but trivial for memo.
        import random
        rng = random.Random(1)
        dims = [rng.randint(1, 50) for _ in range(16)]
        # Just confirm it runs and returns a sane (non-negative, finite) answer --
        # the cross-check against tabulation below is what verifies correctness.
        result = mcm_memo(dims)
        assert result >= 0
        assert result < float("inf")


class TestMCMTabulation:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_matches_known_optimum(self, case):
        assert mcm_tabulation(case["dims"]) == case["expected"]

    def test_single_matrix_costs_nothing(self):
        assert mcm_tabulation([10, 20]) == 0


class TestAllImplementationsAgree:
    @pytest.mark.parametrize("case", EXAMPLES)
    def test_memo_matches_recursive(self, case):
        assert mcm_memo(case["dims"]) == mcm_recursive(case["dims"])

    @pytest.mark.parametrize("case", EXAMPLES)
    def test_tabulation_matches_recursive(self, case):
        assert mcm_tabulation(case["dims"]) == mcm_recursive(case["dims"])

    def test_agree_across_random_chains(self):
        import random
        rng = random.Random(7)
        for _ in range(20):
            n = rng.randint(1, 8)  # keep small enough for mcm_recursive to finish quickly
            dims = [rng.randint(1, 30) for _ in range(n + 1)]
            assert mcm_memo(dims) == mcm_recursive(dims)
            assert mcm_tabulation(dims) == mcm_recursive(dims)


class TestOptimalParenthesization:
    def test_matches_known_classic_result(self):
        cost, parens = optimal_parenthesization(EXAMPLE_CLASSIC["dims"])
        assert cost == EXAMPLE_CLASSIC["expected"]
        # Every matrix must appear exactly once, and the parenthesization must be
        # well-formed (balanced parens) -- not pinned to one exact string, since more
        # than one optimal grouping can exist when costs tie.
        for i in range(1, len(EXAMPLE_CLASSIC["dims"])):
            assert f"A{i}" in parens
        assert parens.count("(") == parens.count(")")

    def test_single_matrix_has_no_parens(self):
        cost, parens = optimal_parenthesization([10, 20])
        assert cost == 0
        assert parens == "A1"

    def test_reported_cost_matches_tabulation(self):
        import random
        rng = random.Random(3)
        for _ in range(10):
            n = rng.randint(1, 10)
            dims = [rng.randint(1, 40) for _ in range(n + 1)]
            cost, _ = optimal_parenthesization(dims)
            assert cost == mcm_tabulation(dims)
